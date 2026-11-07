'use strict';

const { execFileSync } = require('child_process');
const { luauAst } = require('./harness');
const fs = require('fs');

function loadAst(filePath) {
  const exe = luauAst();
  let stdout;
  try {
    stdout = execFileSync(exe, [filePath], { encoding: 'latin1', maxBuffer: 256 * 1024 * 1024 });
  } catch (e) {
    const msg = e.stderr || e.message || '';
    throw new SyntaxError('not valid Luau: ' + msg.trim().split('\n').slice(0, 3).join(' | '));
  }
  return JSON.parse(stdout).root;
}

function loc(node) {
  const [a, b] = node.location.split(' - ');
  const [l1, c1] = a.split(',').map(Number);
  const [l2, c2] = b.split(',').map(Number);
  return [l1, c1, l2, c2];
}

function textOf(lines, node) {
  const [l1, c1, l2, c2] = loc(node);
  if (l1 === l2) return lines[l1].slice(c1, c2);
  return [lines[l1].slice(c1), ...lines.slice(l1 + 1, l2), lines[l2].slice(0, c2)].join('\n');
}

function walkAst(node, fn) {
  if (!node || typeof node !== 'object') return;
  if (Array.isArray(node)) { node.forEach(v => walkAst(v, fn)); return; }
  fn(node);
  Object.values(node).forEach(v => walkAst(v, fn));
}

function localName(expr) {
  if (expr && expr.type === 'AstExprLocal') return expr.local.name;
  return null;
}

function unwrapGroup(e) {
  while (e && e.type === 'AstExprGroup') e = e.expr;
  return e;
}

const OPS = {
  CompareLt: (a, b) => a < b,
  CompareLe: (a, b) => a <= b,
  CompareGt: (a, b) => a > b,
  CompareGe: (a, b) => a >= b,
  CompareEq: (a, b) => a === b,
  CompareNe: (a, b) => a !== b,
};
const FLIP = {
  CompareLt: 'CompareGt', CompareLe: 'CompareGe',
  CompareGt: 'CompareLt', CompareGe: 'CompareLe',
  CompareEq: 'CompareEq', CompareNe: 'CompareNe',
};

function evalCond(cond, varName, value) {
  if (!cond || cond.type !== 'AstExprBinary' || !OPS[cond.op]) return null;
  const l = cond.left, r = cond.right;
  const op = cond.op;
  if (localName(l) === varName && r.type === 'AstExprConstantNumber')
    return OPS[op](value, r.value);
  if (localName(r) === varName && l.type === 'AstExprConstantNumber')
    return OPS[FLIP[op]](value, l.value);
  return null;
}

// v15.0 compiles the dispatch to `while true do local op = arr[pc]; ... end`.
// v15.1 changed two things at once, and both have to be accepted here or nothing
// downstream can work at all:
//
//   1. the loop is `repeat ... until false`, not `while true`;
//   2. the opcode array is not a local — the whole VM state is one table captured as an
//      upvalue, so the read is `UcE[3][n]`, i.e. an index of an index.
//
// Accepting `v.expr` as either a local name or an index of a local name covers both
// without weakening the check: the inner index still has to be a plain local, which is
// the program counter in both families.
function _dispatchArr(v) {
  if (!v || v.type !== 'AstExprIndexExpr') return null;

  const direct = localName(v.expr);
  if (direct) return direct;

  // `T[<n>][<pc>]` — the state table, holding the opcode array at a fixed slot
  if (v.expr.type === 'AstExprIndexExpr') {
    const table = localName(v.expr.expr);
    if (table) return table;
  }

  return null;
}

// `AstStatWhile` with a true condition, or `AstStatRepeat` with `until false`. The
// `until true` repeats that v15.1 sprinkles through its bit32 helpers are NOT dispatch
// loops: they run once by construction, so the form matters.
function _isDispatchLoop(n) {
  if (n.type !== 'AstStatWhile' && n.type !== 'AstStatRepeat') return false;

  const cond = n.condition;
  if (!cond || cond.type !== 'AstExprConstantBool') return false;

  return n.type === 'AstStatWhile' ? cond.value === true : cond.value === false;
}

function findDispatchers(root) {
  const found = [];
  walkAst(root, n => {
    if (!_isDispatchLoop(n)) return;
    const body = n.body.body;
    if (!body || body.length < 2) return;
    const st = body[0];
    if (!['AstStatLocal', 'AstStatAssign'].includes(st.type)) return;
    if ((st.values || []).length !== 1) return;
    let opname;
    if (st.type === 'AstStatLocal') {
      opname = st.vars[0].name;
    } else {
      opname = localName(st.vars[0]);
      if (!opname) return;
    }
    const v = st.values[0];
    const arr = _dispatchArr(v);
    const pc = arr ? localName(v.index) : null;
    if (!arr || !pc || body[1].type !== 'AstStatIf') return;
    found.push({ node: n, op: opname, arr, pc, tree: body[1], rest: body.slice(2) });
  });
  return found;
}

function resolveOp(tree, varName, value) {
  let node = tree;
  while (true) {
    if (!node) return null;
    if (node.type === 'AstStatBlock') {
      if (node.body && node.body.length >= 1 && node.body[0].type === 'AstStatIf' &&
          evalCond(node.body[0].condition, varName, value) !== null) {
        node = node.body[0];
        continue;
      }
      return node;
    }
    if (node.type === 'AstStatIf') {
      const r = evalCond(node.condition, varName, value);
      if (r === null) return node;
      node = r ? node.thenbody : (node.elsebody || null);
      continue;
    }
    return node;
  }
}

function _makerParams(maker) {
  const args = maker.args || [];
  const visible = {};
  args.forEach((a, i) => { visible[a.name] = i; });
  const counts = {}, kcounts = {}, used = new Set();

  // v15.1 holds the whole VM state in one table built at the top of the maker —
  // `local UcE = {S, _, c, B}` — and the dispatcher reads `UcE[3][pc]`. The proto is
  // therefore whichever *parameter* sits in that slot, not a self-indexed one. Mapping
  // the table lets the sampler pick the right argument by name.
  const slotProto = {};
  function nameOf(expr) {
    return expr && expr.type === 'AstExprLocal' ? expr.local.name : null;
  }
  (function collectSlots(n) {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { n.forEach(collectSlots); return; }
    if (n.type === 'AstStatLocal' && (n.vars || []).length === 1) {
      const v = n.vars[0], vals = n.values || [];
      if (vals[0] && vals[0].type === 'AstExprTable') {
        const tableName = v.name;
        // `{S, _, c, B}` is array-style: the key is the 1-based position, and
        // `UcE[3]` is therefore the third item. An explicit `[n] = v` key is used when
        // Luraph writes one, so both spellings map.
        let positional = 0;
        for (const item of vals[0].items || []) {
          positional += 1;
          const key = item.key
            ? (item.key.type === 'AstExprConstantNumber' ? String(item.key.value) : null)
            : String(positional);
          const nm = nameOf(item.value);
          if (tableName && key && nm) slotProto[`${tableName}[${key}]`] = nm;
        }
      }
    }
    Object.values(n).forEach(collectSlots);
  })(maker.body);

  function visit(n) {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { n.forEach(visit); return; }
    if (n.type === 'AstExprLocal') used.add(n.local.location);

    if (n.type === 'AstExprIndexExpr' && n.expr?.type === 'AstExprLocal' &&
        n.index?.type === 'AstExprIndexExpr' && n.index?.expr?.type === 'AstExprLocal' &&
        n.index.expr.local.location === n.expr.local.location) {
      const k = n.expr.local.location;
      counts[k] = (counts[k] || 0) + 1;
    }

    if (n.type === 'AstExprIndexExpr' && n.expr?.type === 'AstExprLocal' &&
        n.index?.type === 'AstExprConstantNumber') {
      const k = n.expr.local.location;
      kcounts[k] = (kcounts[k] || 0) + 1;
    }
    Object.values(n).forEach(visit);
  }
  visit(maker.body);

  // A dispatch read `TBL[<n>][pc]` names the proto directly; prefer it over the
  // heuristic below, which only knows the v15.0 shape.
  let pi = -1;
  (function findDispatchSlot(n) {
    if (pi >= 0 || !n || typeof n !== 'object') return;
    if (Array.isArray(n)) { n.forEach(findDispatchSlot); return; }
    if (n.type === 'AstExprIndexExpr' && n.expr?.type === 'AstExprIndexExpr' &&
        n.expr.expr?.type === 'AstExprLocal' && n.index?.type === 'AstExprLocal') {
      const key = n.expr.index?.type === 'AstExprConstantNumber' ? String(n.expr.index.value) : null;
      const nm = key ? slotProto[`${n.expr.expr.local.name}[${key}]`] : null;
      if (nm && visible[nm] !== undefined) pi = visible[nm];
    }
    Object.values(n).forEach(findDispatchSlot);
  })(maker.body);

  const cands = Object.values(visible).filter(i => i > 0);
  if (pi < 0) {
    pi = cands.length
      ? cands.reduce((best, i) => {
          const bc = counts[args[best]?.location] || 0;
          const ic = counts[args[i]?.location] || 0;
          return ic > bc ? i : (ic === bc && i === 1 ? i : best);
        }, cands[0])
      : 1;
    if (!counts[args[pi]?.location]) {
      pi = 1;
      const kc = cands.filter(i => kcounts[args[i]?.location]);
      if (kc.length) pi = kc.reduce((best, i) => (kcounts[args[i].location] > kcounts[args[best].location] ? i : best), kc[0]);
      else if (kcounts[args[0]?.location] && !args.slice(1).some(a => kcounts[a?.location])) pi = 0;
    }
  }
  const others = Object.values(visible).filter(i => i !== 0 && i !== pi && used.has(args[i]?.location)).sort();
  const ui = others.length ? others[0] : pi + 1;
  return [pi, ui];
}

function closureEntries(root) {
  const dispNodes = findDispatchers(root).map(d => d.node);
  const seen = new Map();

  function walk(n, stack) {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { n.forEach(v => walk(v, stack)); return; }
    if (n.type === 'AstExprFunction') stack = [...stack, n];
    if ((n.type === 'AstStatWhile' || n.type === 'AstStatRepeat') && dispNodes.includes(n)) {
      const inner = [...stack].reverse().find(f => f.vararg && (!f.args || f.args.length === 0));
      const outer = [...stack].reverse().find(f => f.args && f.args.length >= 2);
      if (inner && outer) {
        const firstStmt = inner.body.body[0];
        if (firstStmt) {
          const [l1, c1] = loc(firstStmt);
          const [pi] = _makerParams(outer);
          seen.set(`${l1},${c1}`, { l: l1, c: c1, name: outer.args[pi].name });
        }
      }
    }
    Object.values(n).forEach(v => walk(v, stack));
  }
  walk(root, []);
  return [...seen.values()].map(({ l, c, name }) => [l, c, name]);
}

function _declsIn(fn) {
  const keys = {};
  function visit(n) {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { n.forEach(visit); return; }
    const t = n.type;
    if (t === 'AstExprFunction' && n !== fn) return;
    if (t === 'AstStatLocal') (n.vars || []).forEach(v => { keys[v.location] = v.name; });
    else if (t === 'AstStatLocalFunction') keys[n.name.location] = n.name.name;
    else if (t === 'AstStatFor') keys[n.var.location] = n.var.name;
    else if (t === 'AstStatForIn') (n.vars || []).forEach(v => { keys[v.location] = v.name; });
    Object.values(n).forEach(visit);
  }
  (fn.args || []).forEach(a => { keys[a.location] = a.name; });
  visit(fn.body);
  return keys;
}

function _captures(info) {
  const makerDecls = _declsIn(info.maker);
  const used = {};
  function visit(n) {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { n.forEach(visit); return; }
    if (n.type === 'AstExprLocal') {
      const k = n.local.location;
      if (makerDecls[k]) used[k] = makerDecls[k];
    }
    Object.values(n).forEach(visit);
  }
  visit(info.vm);
  const names = {};
  Object.entries(used).forEach(([k, name]) => { (names[name] = names[name] || []).push(k); });
  const dup = new Set(Object.entries(names).filter(([, ks]) => ks.length > 1).map(([n]) => n));
  const byName = {};
  Object.entries(makerDecls).forEach(([k, nm]) => { (byName[nm] = byName[nm] || []).push(k); });
  return Object.keys(names).filter(nm => !dup.has(nm) && (byName[nm] || []).length === 1).sort();
}

// Does this function body contain (or lead to) one of the dispatch loops? Used to pick
// the VM closure out of a chain of nested functions without assuming a fixed depth.
function _reaches(fn, dispNodes) {
  let hit = false;
  walkAst(fn, n => {
    if (hit) return;
    if (dispNodes.includes(n)) hit = true;
  });
  return hit;
}

function makerInfo(root) {
  const dispNodes = findDispatchers(root).map(d => d.node);
  const out = new Map();

  function walk(n, stack, stmts) {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { n.forEach(v => walk(v, stack, stmts)); return; }
    const t = n.type;
    if (t === 'AstExprFunction') stack = [...stack, n];
    if (t && t.startsWith('AstStat')) stmts = [...stmts, [n, stack.length]];
    if ((t === 'AstStatWhile' || t === 'AstStatRepeat') && dispNodes.includes(n)) {

      let oi = -1;
      for (let i = stack.length - 1; i >= 0; i--) {
        if ((stack[i].args || []).length >= 2) { oi = i; break; }
      }

      // The VM closure is not always the function directly inside the maker. v15.0 puts
      // it there (maker -> closure -> dispatch), but v15.1 inserts a wrapper that only
      // v15.0 puts the VM closure directly inside the maker (stack[oi+1]). v15.1 adds a
      // wrapper that only rebinds the state table, so the closure is one level deeper.
      // Prefer the adjacent slot — that is the shape every working sample has — and only
      // walk further when it does not reach a dispatch loop.
      let clo = null;
      if (stack[oi + 1] && _reaches(stack[oi + 1], dispNodes)) {
        clo = stack[oi + 1];
      } else {
        for (let ci = oi + 1; ci < stack.length && !clo; ci++) {
          if (_reaches(stack[ci], dispNodes)) clo = stack[ci];
        }
      }

      const cloDepth = clo ? stack.indexOf(clo) : -1;

      if (oi >= 0 && clo && cloDepth > oi) {
        // The assigning statement's depth is not fixed: v15.0 writes `local I,a = …` in
        // the maker's body (one level in), v15.1 writes `AuU[121] = …` around it (one
        // level out). Taking the assign whose product *is* the maker function covers
        // both, and falls back to the old fixed offset if nothing matches.
        const makerStmts = stmts.filter(([, d]) => d === cloDepth).map(([s]) => s);
        const enclosed = stmts
          .filter(([s]) => s.type === 'AstStatAssign')
          .filter(([s]) => {
            let hit = false;
            walkAst(s, x => { if (x === stack[oi]) hit = true; });
            return hit;
          });
        const st = (makerStmts[makerStmts.length - 1] && makerStmts[makerStmts.length - 1].type === 'AstStatAssign'
          ? makerStmts[makerStmts.length - 1]
          : null) || (enclosed.length ? enclosed[enclosed.length - 1][0] : null);
        if (st) {
          (st.vars || []).forEach((v, idx) => {
            const e = unwrapGroup((st.values || [])[idx]);
            // v15.1 wraps the entry in one more function, so the value stored is either
            // the dispatch-bearing closure (v15.0) or the maker that produces it.
            if (e !== clo && e !== stack[oi]) return;

            // v15.0 binds the VM closure to a local (`local f = function...`). v15.1
            // stores every entry on one table instead (`AuU[121] = function...`, or
            // `AuU.Pn = ...`), so the target is an index expression and `localName`
            // returns null for it. The slot name is what the hook has to use, so both
            // shapes resolve to a single identifier.
            const target = localName(v) || (v.type === 'AstExprIndexExpr' ? (localName(v.index) || localName(v.expr)) : null);
            if (!target) return;

            const [, , l2, c2] = loc(st);
            const key = `${l2},${c2}`;
            if (!out.has(key)) {
              const [pi, ui] = _makerParams(stack[oi]);
              out.set(key, {
                at: [l2, c2], var: target,
                proto: stack[oi].args[pi].name,
                proto_index: pi, upvals_index: ui,
                pf_key: stack[oi].args[pi].name,
                maker: stack[oi], vm: clo, stmt: st,
              });
            }
          });
        }
      }
    }
    Object.values(n).forEach(v => walk(v, stack, stmts));
  }
  walk(root, [], []);
  const results = [...out.values()];
  results.forEach(info => { info.captures = _captures(info); });
  return results;
}

function patchEntries(source, filePath, chunkTag) {
  const root = loadAst(filePath);
  const lines = source.split('\n');
  const tag = chunkTag;
  const edits = [];

  for (const [l1, c1, name] of closureEntries(root)) {
    const k = `(${name} or __PID)`;
    edits.push([l1, c1,
      `if not __PID[${k}] then __PID.n=__PID.n+1;__PID[${k}]=__PID.n;end;` +
      `__ENT.n=__ENT.n+1;__ENT[__ENT.n%64]=__PID[${k}];__PLAST[__PID[${k}]]=__ENT.n;` +
      `if __SKIPP[__PID[${k}]] then return end;`
    ]);
  }

  for (const info of makerInfo(root)) {
    const [l2, c2] = info.at;
    const pv = info.proto;
    const v = info.var;
    const pfKey = info.pf_key || pv;
    let code = ` __PF[${v}]=${pfKey} `;
    const cap = info.captures.map(nm => `__PA[${pv}].${nm}=${nm};`).join('');
    code += `if __PA and not __PA[${pv}] then __PA[${pv}]={};__PA.n=__PA.n+1;__PA[${pv}].__seq=__PA.n;` +
            `__PA[${pv}].__maker="${tag}@${l2},${c2}";__PK[${pfKey}]=${v};${cap} end `;
    edits.push([l2, c2, code]);
  }

  edits.sort((a, b) => b[0] !== a[0] ? b[0] - a[0] : b[1] - a[1]);
  for (const [l, c, code] of edits) {
    lines[l] = lines[l].slice(0, c) + code + lines[l].slice(c);
  }
  return lines.join('\n');
}

function patchSpin(src) {
  // The dispatch head, in every shape Luraph has used:
  //   v15.0  while true do local m=M[j];
  //   v15.1  repeat local RH,PH=UcE[3][n];      <- a NUMBER slot, then the pc
  // The value is therefore either `NAME[NAME]` or `NAME[<const>][NAME]`. Everything else
  // that looks similar is a helper loop that runs once at startup, so matching the wrong
  // ones is harmless — but NOT matching the real one is fatal: `__SPIN.f()` is never
  // called from the hot path, so the runtime never checks its budget or emits a
  // heartbeat, and the process can only be stopped by the wall clock.
  return src.replace(
    /(?:while true do|repeat) (?:local )?[A-Za-z_]+(?:,[A-Za-z_]+)*=[A-Za-z_]+\[[^\]]+\]\[[A-Za-z_]+\];|(?:while true do|repeat) (?:local )?[A-Za-z_]+(?:,[A-Za-z_]+)*=[A-Za-z_]+\[[A-Za-z_]+\];/g,
    m => m + '__SPIN.n=__SPIN.n+1;if __SPIN.n>=__SPIN.step then __SPIN.f()end;'
  );
}

module.exports = {
  loadAst, loc, textOf, walkAst, localName,
  unwrapGroup,
  findDispatchers, resolveOp, evalCond,
  closureEntries, makerInfo, patchEntries, patchSpin,
};
