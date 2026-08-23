"""
GPU Riemann-Siegel Scanner  -  v6  (INSTRUMENTED with timing splits)
====================================================================
This is v6 with a wall-clock timing harness bolted onto every phase.
Nothing about the math or completeness logic changed. Only additions:

  * A `Timers` helper (context manager + cumulative counters).
  * A per-chunk breakdown printed on the same lines as the progress log,
    plus a full accounting table every 5 chunks.
  * `torch.cuda.synchronize()` fences around GPU work so GPU-side time is
    reported as GPU time, not as "idle CPU while a launch is in flight".
  * Fine-grained splits inside `compute_Z_array` (theta pool.map, anchor
    pool.map, residue pool.map, upload, GPU compute, download, R-interp),
    inside `gpu_main_sum_hp` / `gpu_main_sum` (upload / compute / download,
    aggregated across all tiles), inside `process_chunk` (boundaries,
    nzeros, main sweep, fine re-sweep, extremum loop with sub-splits for
    mp_Z_verify and precise_gap), and in the main loop (append_hits,
    verify_violation, save_ckpt, empty_cache).
  * A cumulative "totals" report on clean exit AND on Ctrl-C.

The report prints, per chunk:
    wall = actual wall time
    accounted = sum of all named timers
    idle = wall - accounted  (this is what you want to drive to zero)

Interpretation reminder: `idle` also captures anything not wrapped in a
timer (Python overhead, GC, small housekeeping), so it will never hit
exactly zero. But large `idle` values -- or large gaps between the sum
of pool.map times and the wall time of that phase -- point at where the
CPU or GPU is sitting on its hands.
"""

import os, json, time, argparse, sys
import numpy as np
from datetime import datetime
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from mpmath import (mp, siegelz, siegeltheta, zeta, mpf, findroot, nzeros,
                    pi as mp_pi, sqrt as mp_sqrt, floor as mp_floor,
                    log as mp_log, arg as mp_arg)

# NOTE: torch is imported lazily (inside GPU functions), NOT at module top.
# On Windows, multiprocessing spawns re-import this module in every worker;
# a top-level `import torch` would make all 36 workers build CUDA contexts
# (commit-charge blowup -> "paging file too small"). Workers only use mpmath.
def _check_torch():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ==========================================================================
#                              TIMING HARNESS
# ==========================================================================
# Lightweight, no external deps. `T.time("name")` context manager captures
# per-phase wall clock; `T.time_gpu("name")` additionally issues cuda.sync
# on entry/exit so GPU-side time is measured, not launch-overhead time.
#
# `T.reset_chunk()` at the start of each chunk gives a per-chunk view;
# `T.chunk` accumulates INSIDE that chunk while `T.totals` accumulates
# across the WHOLE run. Both are updated by the same context manager, so
# every timer contributes to both automatically.
#
# LIVE=True also prints each top-level operation's start/end to the console
# AS IT HAPPENS (with a running chunk-relative timestamp), so you can watch
# progress on multi-minute chunks instead of waiting for the end-of-chunk
# summary. Only depth<=1 phases print live (not the thousands of tiny GPU
# tile timers); the full breakdown still prints at chunk end.
# --------------------------------------------------------------------------
LIVE = True
class Timers:
    """
    Event-logging tracer. Same interface as before (.time / .time_gpu /
    .reset_chunk / .format_chunk / .format_totals) so no call site changes,
    BUT now every enter/exit is recorded in a SEQUENTIAL EVENT LOG with an
    absolute timestamp and nesting depth. This is what exposes time that
    leaks BETWEEN timed sections -- the "it finishes a function then just
    sits" problem. A top-down tally can't show that (nested timers double-
    count); a sequential log can, because any wall-clock gap between one
    event ending and the next beginning is untimed time, shown explicitly.

    Per chunk we record events as (t_rel, kind, name, depth) where t_rel is
    seconds since chunk start, kind is 'EN' (enter) or 'EX' (exit). The
    report walks the log in order and prints, at each transition, how long
    passed -- flagging any interval where NO timer was active (pure gap) or
    where only a leaf timer was active (real work). The GAPS are the leak.
    """
    def __init__(self):
        self.totals = {}   # cumulative seconds by name (whole run)
        self.counts = {}   # how many times each timer fired (whole run)
        self.chunk  = {}   # cumulative seconds by name (current chunk)
        self.log    = []   # sequential events THIS chunk: (t_rel, kind, name, depth)
        self.depth  = 0    # current nesting depth
        self._chunk_t0 = time.perf_counter()

    def _record(self, kind, name):
        self.log.append((time.perf_counter() - self._chunk_t0, kind, name, self.depth))

    def _live(self, kind, name, dt=None):
        # Live console print of an operation's start/end. Gated on LIVE and on
        # depth so we don't spam the thousands of tiny GPU-tile timers. Only
        # top-level chunk phases (depth 0-1) print live; the full detail is in
        # the end-of-chunk summary.
        if not LIVE or self.depth > 1:
            return
        t_rel = time.perf_counter() - self._chunk_t0
        pad = "  " * self.depth
        if kind == 'EN':
            print(f"    [{t_rel:7.1f}s] >> {pad}{name}", flush=True)
        else:
            print(f"    [{t_rel:7.1f}s] << {pad}{name}  ({dt:.2f}s)", flush=True)

    @contextmanager
    def time(self, name):
        self._live('EN', name)
        self._record('EN', name); self.depth += 1
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.depth -= 1; self._record('EX', name)
            self._live('EX', name, dt)
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1
            self.chunk[name]  = self.chunk.get(name, 0.0) + dt

    @contextmanager
    def time_gpu(self, name):
        """Same as time() but syncs CUDA before/after so the reported time
        reflects actual GPU work, not just kernel launch return. Import is
        lazy so mpmath-only worker processes don't touch torch."""
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False
        if has_cuda:
            import torch; torch.cuda.synchronize()
        self._live('EN', name)
        self._record('EN', name); self.depth += 1
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if has_cuda:
                import torch; torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.depth -= 1; self._record('EX', name)
            self._live('EX', name, dt)
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1
            self.chunk[name]  = self.chunk.get(name, 0.0) + dt

    def reset_chunk(self):
        self.chunk = {}
        self.log = []
        self.depth = 0
        self._chunk_t0 = time.perf_counter()

    def format_chunk(self, wall):
        """
        SEQUENTIAL trace + gap analysis for the just-completed chunk.

        We reconstruct, from the event log, the intervals during which the
        DEEPEST active timer was some leaf (real measured work) vs intervals
        during which NO timer was active at all (pure untimed gap -- the leak).

        The key output is the GAP LIST: every stretch of wall time where the
        active-timer depth was 0 (nothing timed) or where a parent timer was
        active but no child -- i.e. time spent inside a timed function that is
        NOT covered by any inner timer. Both are 'unaccounted at this level'
        and are exactly where the process 'sits and sits'.
        """
        lines = [f"  chunk timing (wall={wall:.3f}s):"]
        events = self.log

        # ---- 1) GAP DETECTION: for every interval between two consecutive
        # events, determine the DEEPEST currently-open timer. If that timer is
        # a PARENT (i.e. the interval sits inside it but no child timer is
        # running), that time is 'unattributed within <parent>' -- the leak.
        # We attribute each interval to whatever the innermost open timer is;
        # intervals with NO open timer are pure gaps between chunks.
        open_stack = []                 # names currently open, innermost last
        unattributed = {}               # innermost_name -> seconds not in any child
        pure_gap = 0.0
        for i in range(len(events)):
            t_rel, kind, name, depth = events[i]
            if i > 0:
                interval = t_rel - events[i-1][0]
                if interval > 0:
                    if open_stack:
                        # time inside innermost open timer, not in any child
                        inner = open_stack[-1]
                        unattributed[inner] = unattributed.get(inner, 0.0) + interval
                    else:
                        pure_gap += interval
            if kind == 'EN':
                open_stack.append(name)
            else:
                # pop matching (innermost) name
                for j in range(len(open_stack)-1, -1, -1):
                    if open_stack[j] == name:
                        open_stack.pop(j); break

        # ---- 2) compact sequential trace at depth<=1 (shows order + when) ----
        lines.append("  --- sequential trace (t_rel: EN/EX @depth<=1) ---")
        for (t_rel, kind, name, depth) in events:
            if depth <= 1:
                lines.append(f"    {t_rel:8.2f}s  {kind}  {'  '*depth}{name}")

        # ---- 3) THE LEAK: unattributed time inside each parent, and pure gaps.
        # 'unattributed inside <parent>' = time spent in that function NOT
        # covered by any inner timer = the 'sits and sits' time. Sorted big
        # first. This is what to drive to zero (wrap it or explain it).
        lines.append("  --- UNATTRIBUTED time (inside a timer, no child running) ---")
        ua = sorted(unattributed.items(), key=lambda x: -x[1])
        ua_total = 0.0
        for k, v in ua:
            if v > 0.3:
                ua_total += v
                lines.append(f"    inside {k:32s} {v:8.2f}s  {100.0*v/max(wall,1e-9):5.1f}%")
        if pure_gap > 0.3:
            lines.append(f"    {'PURE GAP (between chunks/untimed)':38s} {pure_gap:8.2f}s  "
                         f"{100.0*pure_gap/max(wall,1e-9):5.1f}%")
        lines.append(f"    >>> total unattributed+gap: {ua_total+pure_gap:.2f}s "
                     f"({100.0*(ua_total+pure_gap)/max(wall,1e-9):.1f}% of wall) "
                     f"<-- this is the leak")

        # ---- 4) leaf-timer tally (non-nested view) ----
        # A leaf timer is one that never had a child EN before its EX. Compute
        # leaf durations by pairing EN/EX and checking if any event occurred
        # between them at greater depth.
        leaf_time = {}
        stack = []
        for (t_rel, kind, name, depth) in events:
            if kind == 'EN':
                stack.append([name, t_rel, depth, False])  # False=has no child yet
                if len(stack) >= 2:
                    stack[-2][3] = True                     # parent now has a child
            else:  # EX
                for j in range(len(stack)-1, -1, -1):
                    if stack[j][0] == name:
                        nm, t_en, dp, had_child = stack.pop(j)
                        if not had_child:                   # leaf
                            leaf_time[nm] = leaf_time.get(nm, 0.0) + (t_rel - t_en)
                        break
        lines.append("  --- LEAF timer time (non-overlapping real work) ---")
        leaf_items = sorted(leaf_time.items(), key=lambda x: -x[1])
        leaf_total = sum(v for _, v in leaf_items)
        for k, v in leaf_items:
            if v > 0.05:
                lines.append(f"    {k:34s} {v:7.2f}s  {100.0*v/max(wall,1e-9):5.1f}%")
        lines.append(f"    {'== leaf total (real work) ==':34s} {leaf_total:7.2f}s  "
                     f"{100.0*leaf_total/max(wall,1e-9):5.1f}%")
        lines.append(f"    {'== wall - leaf (gaps+overhead) ==':34s} "
                     f"{wall-leaf_total:7.2f}s  {100.0*(wall-leaf_total)/max(wall,1e-9):5.1f}%")
        return "\n".join(lines)

    def format_totals(self):
        items = sorted(self.totals.items(), key=lambda x: -x[1])
        total = sum(v for _, v in items)
        lines = ["", "=" * 78, "CUMULATIVE TIMING TOTALS (whole run)", "=" * 78]
        lines.append(f"  {'phase':34s} {'total (s)':>10s} {'count':>7s} {'avg (ms)':>10s} {'share':>7s}")
        lines.append("  " + "-" * 74)
        for k, v in items:
            n = self.counts[k]
            avg_ms = v / max(n, 1) * 1000.0
            share = 100.0 * v / max(total, 1e-9)
            lines.append(f"  {k:34s} {v:10.2f} {n:7d} {avg_ms:10.2f} {share:6.1f}%")
        lines.append("=" * 78)
        return "\n".join(lines)

T = Timers()   # module-level singleton; workers do NOT see this


# ---------------- CONFIG ----------------
# (unchanged from v6)
VALIDATE = False

if VALIDATE:
    T_BASE      = 6990.0
    N_CHUNKS    = 5
    CHECKPOINT  = "zeta_validate_checkpoint.json"
    RESULTS_LOG = "zeta_validate_hits.csv"
    COUNT_LOG   = "zeta_validate_count.csv"
    ZEROS_LOG   = "zeta_validate_zeros.csv"
    _VALIDATE_CHUNK_T = 100.0    # validation uses the ORIGINAL 100-unit chunk
                                 # so results are directly comparable to the
                                 # known-good 562/562 run. The big CHUNK_T below
                                 # is a sweep-speed choice, not a validation one.
else:
    T_BASE      = 1e13
    N_CHUNKS    = 100
    CHECKPOINT  = "zeta_scan_v6_1p0e13.json"
    RESULTS_LOG = "zeta_hits_v6_1p0e13.csv"
    COUNT_LOG   = "zeta_count_v6_1p0e13.csv"
    ZEROS_LOG   = "zeta_zeros_v6_1p0e13.csv"

CHUNK_T        = 1000.0   # bigger chunks amortize the per-chunk single-thread
                          # nzeros cost (~37s at 1e12, fixed per chunk) over 10x
                          # more work. Profiling showed nzeros_b is the visible
                          # "idle" (1 core grinding, 35 dark). Widening the chunk
                          # dilutes it directly. RAM is not a constraint (256GB);
                          # GPU stays batched by GPU_MAX_MATRIX.
if VALIDATE:
    CHUNK_T = _VALIDATE_CHUNK_T   # validation runs at the original 100-unit
                                  # chunk for direct comparability to 562/562.
STEP           = 0.01
FINE_STEP      = 0.001
LEHMER_THR     = 0.02
NGAP_REFINE    = 0.05
DPS_FILTER     = 30
DPS_VERIFY     = 50
ANCHOR_SPACING = 25.0    # R(t) is dead-linear over a chunk (~2.7e-9 per 10u at
                         # 1e12), so few anchors interpolate it exactly. Each
                         # anchor is a full ~9s mpmath RS sum -> profiling showed
                         # pool_map_anchors ~56s/chunk was the largest cost. At
                         # CHUNK_T=1000 this still gives ~40 anchors (plenty to
                         # feed 36 cores; no idle-core problem), cutting the
                         # anchor cost ~6x with interp error ~1e-9 (negligible).
GPU_TILE_N     = 300_000
GPU_MAX_MATRIX = 40_000_000
HP_ARG         = True
N_WORKERS      = None
MP_WORKERS     = 36


# ==========================================================================
#         JSON PROGRESS PROTOCOL + PAUSE FLAG (for UI / distributed)
# ==========================================================================
# Two OPT-IN features that expose the scanner to external supervision without
# changing any of the math or file formats. Default behavior (running the
# script bare) is byte-identical to before -- these only activate when the
# corresponding CLI flags are passed.
#
#   --json-progress
#       In addition to the existing pretty prints, emit one JSON dict per
#       line to a designated stream (default: stdout, tag-prefixed with
#       "@@STATUS@@ " so a UI can grep it out of the mixed live-timer stream).
#       Events: run_start, chunk_start, chunk_end, violation_survived,
#       paused, run_end. A UI parses these to update status widgets.
#
#   --pause-flag PATH
#       Between chunks, check if PATH exists. If yes: save checkpoint (already
#       done every chunk anyway), emit {"event":"paused","next_chunk":N},
#       exit cleanly (code 0). Resume = remove the flag and re-launch the
#       script; the existing checkpoint resume path picks up exactly where
#       it stopped. Zero data loss, zero mid-chunk interruption -- the pause
#       waits for the current chunk to finish, so completeness is preserved.
#
#   --work-unit T0 DT_START DT_END OUTPUT_PREFIX
#       Distributed-worker mode. Run exactly ONE work unit (one chunk from
#       DT_START to DT_END at chunk-base T0), write results with OUTPUT_PREFIX
#       (files: {prefix}_hits.csv, {prefix}_zeros.csv, {prefix}_checkpoint.json,
#       {prefix}_count.csv), then exit. A future coordinator (GIMPS-style)
#       calls the same binary with this flag to farm chunks across contributors.
#       Not yet used but designed in so the scanner IS its own distributed
#       worker without any coordinator existing.
#
# All three flags are additive -- passing --json-progress alone just adds
# structured output; --pause-flag alone just enables cooperative pause; etc.

_JSON_PROGRESS = False           # set by CLI parse; default OFF
_PAUSE_FLAG_PATH = None          # set by CLI parse; default OFF
_STATUS_TAG = "@@STATUS@@ "      # UI parses lines starting with this tag

def _emit_status(event, **kwargs):
    """Emit a structured status event. No-op when --json-progress is off, so
    the default CLI behavior is unchanged. When on, prints one JSON line to
    stdout with a well-known tag prefix so a UI can filter status events out
    of the mixed live-timer stream without parsing every line."""
    if not _JSON_PROGRESS:
        return
    payload = {"event": event, "ts": time.time(), **kwargs}
    print(_STATUS_TAG + json.dumps(payload), flush=True)

def _check_pause_flag():
    """Return True if the pause flag file exists (user requested pause).
    Called BETWEEN chunks so a pause never interrupts a partial computation
    -- the current chunk finishes, its checkpoint saves, then we exit clean."""
    return _PAUSE_FLAG_PATH is not None and os.path.exists(_PAUSE_FLAG_PATH)


# ---------------- exact-t coordinate helper (Stage 3 grid) ----------------
def exact_t(t0_str, dt):
    if isinstance(dt, mpf):
        return mpf(t0_str) + dt
    return mpf(t0_str) + mpf(repr(float(dt)))

# ---------------- mpmath workers ----------------

def _worker_init():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def _R_anchor_worker(args):
    t0_str, dt = args
    mp.dps = DPS_FILTER
    tt = exact_t(t0_str, dt); th = siegeltheta(tt)
    N = int(mp_floor(mp_sqrt(tt/(2*mp_pi))))
    s = mpf(0)
    for n in range(1, N+1):
        s += mp.cos(th - tt*mp_log(n))/mp_sqrt(mpf(n))
    return (dt, float(th), N, float(siegelz(tt) - 2*s))

def _theta_worker(args):
    t0_str, dt = args
    mp.dps = DPS_FILTER
    tt = exact_t(t0_str, dt)
    th = siegeltheta(tt)
    th_mod = float(th % (2*mp_pi))
    return (float(th), int(mp_floor(mp_sqrt(tt/(2*mp_pi)))), th_mod)

def mp_Z_verify(t0_str, dt, dps=DPS_VERIFY):
    mp.dps = dps
    return float(siegelz(exact_t(t0_str, dt)))


def _siegelz_batch_worker(args):
    """Evaluate siegelz at a batch of dt offsets (for parallel gap scan).
    args = (t0_str, dt_list, dps). Returns list of float Z values, in order."""
    t0_str, dt_list, dps = args
    mp.dps = dps
    return [float(siegelz(exact_t(t0_str, d))) for d in dt_list]

def _nzeros_worker(args):
    """Evaluate mpmath.nzeros(t0 + dt) at the given precision. Designed to
    be submitted async so it runs concurrently with the GPU compute_Z_array.
    args = (t0_str, dt, dps). Returns int.

    Why a separate worker (not the main pool): the main 36-worker pool is
    dedicated to CPU-bound parallel batches (theta anchors, precise_gap
    scans). If we submitted nzeros there it would fight the extremum-loop
    parallelization later. A tiny dedicated pool of 1-2 workers just for
    nzeros keeps the two workloads cleanly separated."""
    t0_str, dt, dps = args
    mp.dps = dps
    return int(nzeros(exact_t(t0_str, dt)))

def mean_gap(t):
    return float(2*mp_pi / mp_log(mpf(str(t))/(2*mp_pi)))


# ---------------- high-precision argument reduction (Stage 3) ----------------

def _residue_worker(args):
    t0_str, n_lo, n_hi, dps = args
    mp.dps = dps
    t0 = mpf(t0_str)
    two_pi = 2*mp_pi
    out = np.empty(n_hi - n_lo + 1, dtype=np.float64)
    lnout = np.empty(n_hi - n_lo + 1, dtype=np.float64)
    for i, n in enumerate(range(n_lo, n_hi+1)):
        ln_n = mp_log(n)
        out[i] = float((t0 * ln_n) % two_pi)
        lnout[i] = float(ln_n)
    return (n_lo, out, lnout)

def compute_residues(t0_str, N, pool, dps=40, batch=40000):
    """Parallel r_n = (t0*ln n) mod 2pi and ln(n) for n=1..N.  TIMED SPLITS:
       residues.build_tasks | residues.pool_map | residues.stitch"""
    with T.time("residues.build_tasks"):
        tasks = []
        n = 1
        while n <= N:
            hi = min(n+batch-1, N)
            tasks.append((t0_str, n, hi, dps))
            n = hi + 1
    with T.time("residues.pool_map"):
        results = pool.map(_residue_worker, tasks)
    with T.time("residues.stitch"):
        r_n = np.empty(N+1, dtype=np.float64)
        ln_n = np.empty(N+1, dtype=np.float64)
        r_n[0] = 0.0; ln_n[0] = 0.0
        for n_lo, out, lnout in results:
            r_n[n_lo:n_lo+len(out)] = out
            ln_n[n_lo:n_lo+len(lnout)] = lnout
    return r_n, ln_n

def mpf_to_str(x):
    return mp.nstr(mpf(x), 30) if not isinstance(x, str) else x


def S_of_T_unwound(T_, t_ref, S_ref, dps=DPS_VERIFY):
    mp.dps = dps
    return float(mp_arg(zeta(mpf('0.5') + 1j*mpf('%.6f' % T_)))/mp_pi)

def N_of_T(T_, t_ref, S_ref, dps=DPS_VERIFY):
    mp.dps = dps
    th = float(siegeltheta(mpf('%.6f' % T_)))
    S = S_of_T_unwound(T_, t_ref, S_ref, dps=dps)
    return (th/float(mp_pi) + 1.0 + S, S)


def safe_boundary(t0_str, dt_target, dps=DPS_VERIFY, search=0.08, step=0.004, want=0.20):
    mp.dps = dps
    f = lambda d: abs(float(siegelz(exact_t(t0_str, d))))
    best_d, best_v = dt_target, f(dt_target)
    k = 1
    maxk = int(search/step)
    while k <= maxk:
        for dd in (dt_target + k*step, dt_target - k*step):
            v = f(dd)
            if v > best_v:
                best_v, best_d = v, dd
            if v >= want:
                return dd
        k += 1
    return best_d

def precise_gap(t0_str, dt_ext, pool=None, dps=DPS_VERIFY, reach=0.05,
                fine=0.0005, verbose=False):
    """
    Find the two zeros bracketing the extremum at (t0+dt_ext), return their
    gap. PARALLEL scan: the search window [dt_ext-reach, dt_ext+reach] is
    evaluated at 'fine' resolution across ALL pool workers in one map (this
    was the single-threaded bottleneck -- ~200 serial siegelz calls at 1e12,
    up to ~220s per tight hit). We then locate the sign changes on each side
    of dt_ext from the returned grid and bisect each bracket to full precision.
    Bisection stays serial but is only ~60 cheap evals per side.
    If pool is None, falls back to a serial scan (used only if called without
    a pool).
    """
    mp.dps = dps
    f = lambda d: siegelz(exact_t(t0_str, d))

    # Build the full scan grid around dt_ext.
    npts = int(reach/fine)
    left_ds  = [dt_ext - k*fine for k in range(npts+1)]   # k=0..npts (incl ext)
    right_ds = [dt_ext + k*fine for k in range(npts+1)]

    if pool is not None:
        # Evaluate both wings in parallel across workers. Chunk the points so
        # each worker gets a contiguous batch (amortize dispatch).
        all_ds = left_ds + right_ds
        nw = max(1, cpu_count())
        batch = max(1, len(all_ds)//nw + 1)
        tasks = [(t0_str, all_ds[i:i+batch], dps)
                 for i in range(0, len(all_ds), batch)]
        results = pool.map(_siegelz_batch_worker, tasks)
        flat = [z for sub in results for z in sub]
        left_z  = flat[:len(left_ds)]
        right_z = flat[len(left_ds):]
    else:
        left_z  = [float(f(d)) for d in left_ds]
        right_z = [float(f(d)) for d in right_ds]

    def refine_parallel(d_lo, d_hi):
        """Refine a sign-change bracket [d_lo,d_hi] to a root using PARALLEL
        grid rounds instead of 60 serial siegelz calls (which at 1e12 cost
        ~1.1s each = ~66s per bracket -- the real precise_gap bottleneck).
        Each round evaluates GRID points across the bracket in one pool.map
        and zooms into the sub-interval holding the sign change. GRID=49 pts
        shrinks the bracket ~48x/round; 4 rounds -> ~0.0005/48^4 ~ 1e-11,
        well past the ~1e-9 the gap needs. If no pool, falls back to serial."""
        GRID = 49
        z_lo = float(f(d_lo))
        for _rnd in range(4):
            ds = [d_lo + (d_hi - d_lo)*k/(GRID-1) for k in range(GRID)]
            if pool is not None:
                nw = max(1, cpu_count())
                bs = max(1, len(ds)//nw + 1)
                tasks = [(t0_str, ds[i:i+bs], dps) for i in range(0, len(ds), bs)]
                zs = [z for sub in pool.map(_siegelz_batch_worker, tasks) for z in sub]
            else:
                zs = [float(f(d)) for d in ds]
            # find first sub-interval with a sign change
            found_sub = False
            for k in range(1, GRID):
                if (zs[k-1] >= 0) != (zs[k] >= 0):
                    d_lo, d_hi = ds[k-1], ds[k]
                    z_lo = zs[k-1]
                    found_sub = True
                    break
            if not found_sub:
                break
            if (d_hi - d_lo) < 1e-13:
                break
        return 0.5*(d_lo + d_hi)

    def first_crossing(ds, zs):
        # ds[0] is dt_ext; find first index where sign flips from ds[0], then
        # refine that bracket in parallel.
        z0 = zs[0]
        for i in range(1, len(zs)):
            if (z0 >= 0) != (zs[i] >= 0):
                d_lo = min(ds[i-1], ds[i]); d_hi = max(ds[i-1], ds[i])
                return refine_parallel(d_lo, d_hi)
            z0 = zs[i]
        return None

    lo = first_crossing(left_ds, left_z)
    hi = first_crossing(right_ds, right_z)
    if lo is None or hi is None:
        if verbose:
            print(f"    [gap] no bracket (lo={lo}, hi={hi})")
        return None, None, None
    if verbose:
        print(f"    [gap] zero1 t={float(exact_t(t0_str,lo)):.10f}  "
              f"zero2 t={float(exact_t(t0_str,hi)):.10f}  gap={hi-lo:.10e}")
    return (hi-lo), lo, hi

# ---------------- GPU main sum ----------------
def gpu_main_sum(t_arr, theta_arr, N_arr, tile_n=GPU_TILE_N):
    """Legacy (float64 t) path. Timed splits:
         gpu.legacy.upload | gpu.legacy.compute | gpu.legacy.download
       aggregated over ALL tile iterations for the whole call."""
    import torch
    dev = torch.device('cuda'); dt = torch.float64
    T_total = len(t_arr)
    out = np.empty(T_total, dtype=np.float64)
    t_batch = max(1, min(T_total, GPU_MAX_MATRIX // max(1, tile_n)))

    for b0 in range(0, T_total, t_batch):
        b1 = min(b0 + t_batch, T_total)
        with T.time_gpu("gpu.legacy.upload"):
            t  = torch.tensor(t_arr[b0:b1],     dtype=dt, device=dev)
            th = torch.tensor(theta_arr[b0:b1], dtype=dt, device=dev)
            Nt = torch.tensor(N_arr[b0:b1],     dtype=torch.int64, device=dev)
        Tb = t.shape[0]; Nmax = int(Nt.max().item())
        with T.time_gpu("gpu.legacy.compute"):
            main = torch.zeros(Tb, dtype=dt, device=dev)
            start = 1
            while start <= Nmax:
                stop = min(start+tile_n-1, Nmax)
                n = torch.arange(start, stop+1, dtype=dt, device=dev)
                logn = torch.log(n); inv = 1.0/torch.sqrt(n)
                arg = th.unsqueeze(1) - t.unsqueeze(1)*logn.unsqueeze(0)
                contrib = torch.cos(arg)*inv.unsqueeze(0)
                mask = (n.unsqueeze(0) <= Nt.unsqueeze(1).to(dt))
                main += (contrib*mask).sum(dim=1)
                del n, logn, inv, arg, contrib, mask
                start = stop+1
        with T.time_gpu("gpu.legacy.download"):
            out[b0:b1] = (2.0*main).detach().cpu().numpy()
        del t, th, Nt, main
    return out


def gpu_main_sum_hp(dt_arr, theta_mod_arr, N_arr, r_n, ln_n, t0,
                    tile_n=GPU_TILE_N):
    """High-precision-argument path. Timed splits (aggregated across ALL
    tile iterations of this call):
      gpu.hp.upload_res    -- one-time upload of r_n, ln_n (per call)
      gpu.hp.upload_batch  -- per t-batch upload of theta/N/dt
      gpu.hp.compute       -- kernel time (cuda-synced)
      gpu.hp.download      -- device->host of the result slice"""
    import torch
    dev = torch.device('cuda'); dt = torch.float64
    T_total = len(dt_arr)
    out = np.empty(T_total, dtype=np.float64)
    t_batch = max(1, min(T_total, GPU_MAX_MATRIX // max(1, tile_n)))

    with T.time_gpu("gpu.hp.upload_res"):
        r_n_t  = torch.tensor(r_n,  dtype=dt, device=dev)
        ln_n_t = torch.tensor(ln_n, dtype=dt, device=dev)

    for b0 in range(0, T_total, t_batch):
        b1 = min(b0 + t_batch, T_total)
        with T.time_gpu("gpu.hp.upload_batch"):
            thm = torch.tensor(theta_mod_arr[b0:b1],dtype=dt, device=dev)
            Nt  = torch.tensor(N_arr[b0:b1],         dtype=torch.int64, device=dev)
            dt_ = torch.tensor(dt_arr[b0:b1],        dtype=dt, device=dev)
        Tb = dt_.shape[0]; Nmax = int(Nt.max().item())
        with T.time_gpu("gpu.hp.compute"):
            main = torch.zeros(Tb, dtype=dt, device=dev)
            start = 1
            while start <= Nmax:
                stop = min(start+tile_n-1, Nmax)
                idx = torch.arange(start, stop+1, device=dev)
                rn  = r_n_t[idx]
                lnn = ln_n_t[idx]
                inv = torch.exp(-0.5*lnn)
                arg = thm.unsqueeze(1) - rn.unsqueeze(0) - dt_.unsqueeze(1)*lnn.unsqueeze(0)
                contrib = torch.cos(arg)*inv.unsqueeze(0)
                mask = (idx.unsqueeze(0) <= Nt.unsqueeze(1))
                main += (contrib*mask).sum(dim=1)
                del idx, rn, lnn, inv, arg, contrib, mask
                start = stop+1
        with T.time_gpu("gpu.hp.download"):
            out[b0:b1] = (2.0*main).detach().cpu().numpy()
        del thm, Nt, dt_, main
    del r_n_t, ln_n_t
    return out


def compute_Z_array(t0_str, dt_arr, pool, hp=True):
    """Timed splits (each is one phase within one Z-array build):
         zbuild.build_theta_args    | zbuild.pool_map_theta
         zbuild.pack_theta_arrays   | zbuild.build_anchor_args
         zbuild.pool_map_anchors    | zbuild.anchor_interp
         zbuild.residues            (compute_residues has its own inner splits)
         zbuild.gpu_main_sum_hp     (gpu_main_sum_hp has its own inner splits)
         zbuild.final_add"""
    t0 = float(mpf(t0_str))

    with T.time("zbuild.build_theta_args"):
        args = [(t0_str, float(dt)) for dt in dt_arr]
    with T.time("zbuild.pool_map_theta"):
        th_N = pool.map(_theta_worker, args,
                        chunksize=max(1, len(args)//(4*cpu_count())))
    with T.time("zbuild.pack_theta_arrays"):
        theta_arr = np.array([x[0] for x in th_N])
        N_arr = np.array([x[1] for x in th_N], dtype=np.int64)
        theta_mod_arr = np.array([x[2] for x in th_N])

    da = dt_arr[0]; db = dt_arr[-1]
    n_anchors = max(4, int((db-da)/ANCHOR_SPACING))
    anchor_dt = np.linspace(da, db, n_anchors)

    with T.time("zbuild.build_anchor_args"):
        anchor_args = [(t0_str, float(d)) for d in anchor_dt]
    with T.time("zbuild.pool_map_anchors"):
        ar = pool.map(_R_anchor_worker, anchor_args)
    with T.time("zbuild.anchor_interp"):
        ad = np.array([a[0] for a in ar]); aR = np.array([a[3] for a in ar])
        o = np.argsort(ad)
        R_arr = np.interp(dt_arr, ad[o], aR[o])

    Nmax = int(N_arr.max())
    with T.time("zbuild.residues"):
        r_n, ln_n = compute_residues(t0_str, Nmax, pool)
    with T.time("zbuild.gpu_main_sum_hp"):
        main = gpu_main_sum_hp(dt_arr, theta_mod_arr, N_arr, r_n, ln_n, t0)
    with T.time("zbuild.final_add"):
        result = main + R_arr
    return result

def count_sign_changes(Z):
    return int(np.sum((Z[:-1] >= 0) != (Z[1:] >= 0)))


def verify_violation(t0_str, dt_approx, kind, dps=100):
    mp.dps = dps
    f = lambda d: siegelz(exact_t(t0_str, d))
    want_min = (kind == 'pos_local_min')
    span = 0.05; fine = 0.0005
    ts = [dt_approx + k*fine for k in range(-int(span/fine), int(span/fine)+1)]
    zs = [float(f(t)) for t in ts]
    if want_min:
        j = min(range(len(zs)), key=lambda k: zs[k])
    else:
        j = max(range(len(zs)), key=lambda k: zs[k])
    a, b = ts[max(0,j-1)], ts[min(len(ts)-1,j+1)]
    gr = (5**0.5 - 1)/2
    def g(x):
        v = float(f(x)); return v if want_min else -v
    c = b - gr*(b-a); d = a + gr*(b-a); fc, fd = g(c), g(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc; c = b - gr*(b-a); fc = g(c)
        else:
            a, c, fc = c, d, fd; d = a + gr*(b-a); fd = g(d)
        if abs(b-a) < 1e-11: break
    dt_true = 0.5*(a+b); Z_true = float(f(dt_true))
    tol = 1e-6
    if want_min:
        is_viol = (Z_true > tol)
    else:
        is_viol = (Z_true < -tol)
    return is_viol, dt_true, Z_true

# ---------------- chunk ----------------
def process_chunk(t0_str, dt_start, dt_end, step, pool, dt_a=None, nz_ta=None,
                  nz_pool=None, fine_step=None):
    """Timed splits inside a chunk:
        chunk.safe_boundary_a  | chunk.safe_boundary_b
        chunk.build_dt_grid    | chunk.compute_Z_array (already broken out)
        chunk.nzeros_a         | chunk.nzeros_b
        chunk.count_signs
        chunk.fine_resweep     (only fires on undercount)
        chunk.count_log_write
        chunk.mean_gap
        chunk.extremum_loop           -- outer loop bookkeeping
          chunk.ext.parabola          -- vertex+curvature per extremum
          chunk.ext.mp_Z_verify       -- mpmath verify at extremum
          chunk.ext.precise_gap       -- when ngap_est < NGAP_REFINE

    nz_pool (optional): a small dedicated multiprocessing.Pool for running
    mpmath.nzeros() concurrently with compute_Z_array. Kicked off right
    after safe_boundary_b so the GPU work and the single-threaded nzeros
    computation overlap -- saving roughly min(GPU_time, nzeros_time) per
    chunk on the wall clock. If None, nzeros runs sequentially (backward
    compatible)."""
    with T.time("chunk.safe_boundary_a"):
        if dt_a is None:
            dt_a = safe_boundary(t0_str, dt_start)
    with T.time("chunk.safe_boundary_b"):
        dt_b = safe_boundary(t0_str, dt_end)

    # ---- Kick off nzeros async NOW (before compute_Z_array), so it runs
    # concurrently with the GPU work. Both boundaries are known at this
    # point, so both async submissions can go. On chunks 2+ nz_ta comes in
    # via the seam and we only need to compute nz_tb; on chunk 0 we compute
    # both. We resolve them (call .get()) further down, AFTER compute_Z_array
    # has done its work. If nzeros finishes first the .get() returns
    # instantly; if compute_Z_array finishes first we pay the remaining
    # nzeros time -- either way, net wall time is min(gpu, nzeros) saved
    # vs the old sequential order. If no nz_pool is provided we fall back
    # to sequential (backward compatible).
    nz_ta_async = None
    nz_tb_async = None
    if nz_pool is not None:
        if nz_ta is None:
            nz_ta_async = nz_pool.apply_async(
                _nzeros_worker, ((t0_str, dt_a, DPS_VERIFY),))
        nz_tb_async = nz_pool.apply_async(
            _nzeros_worker, ((t0_str, dt_b, DPS_VERIFY),))

    with T.time("chunk.build_dt_grid"):
        dt_arr = np.arange(dt_a, dt_b, step)
        if dt_arr[-1] < dt_b:
            dt_arr = np.append(dt_arr, dt_b)

    with T.time("chunk.compute_Z_array"):
        Z = compute_Z_array(t0_str, dt_arr, pool)

    # ---- Resolve async nzeros results (or fall back to sequential). The
    # timers still fire and measure the RESIDUAL wait time -- if nzeros
    # finished during compute_Z_array, these will show ~0s each; if it
    # didn't, they show how much of nzeros' cost didn't get hidden by
    # the overlap. Either number is informative for tuning.
    mp.dps = DPS_VERIFY
    if nz_ta is None:
        with T.time("chunk.nzeros_a"):
            if nz_ta_async is not None:
                nz_ta = nz_ta_async.get()
            else:
                nz_ta = int(nzeros(exact_t(t0_str, dt_a)))
    with T.time("chunk.nzeros_b"):
        if nz_tb_async is not None:
            nz_tb = nz_tb_async.get()
        else:
            nz_tb = int(nzeros(exact_t(t0_str, dt_b)))
    required = nz_tb - nz_ta

    with T.time("chunk.count_signs"):
        found = count_sign_changes(Z)

    refined = False
    # fine_step defaults to the module-level FINE_STEP (0.001, 10x denser than
    # the coarse grid). Callers can pass a smaller value (e.g. FINE_STEP/10 =
    # 0.0001) to force an ultra-fine resweep -- used when a chunk's shortfall
    # persists across the next chunk boundary and we want to escalate before
    # concluding it's a real miss.
    _fs = fine_step if fine_step is not None else FINE_STEP
    if found < required:
        with T.time("chunk.fine_resweep"):
            dt_fine = np.arange(dt_a, dt_b, _fs)
            if dt_fine[-1] < dt_b:
                dt_fine = np.append(dt_fine, dt_b)
            Z_fine = compute_Z_array(t0_str, dt_fine, pool)
            found_fine = count_sign_changes(Z_fine)
        refined = True
        if found_fine >= found:
            dt_arr, Z, step, found = dt_fine, Z_fine, _fs, found_fine
        if found_fine < required:
            with T.time("chunk.count_log_write"):
                t_a = float(exact_t(t0_str, dt_a)); t_b = float(exact_t(t0_str, dt_b))
                with open(COUNT_LOG, 'a') as f:
                    f.write(f"{t_a:.4f},{t_b:.4f},{required},{found_fine},"
                            f"{required-found_fine}\n")

    hits, violations = [], []
    dt_mid = 0.5*(dt_start+dt_end)
    with T.time("chunk.mean_gap"):
        mg = mean_gap(float(exact_t(t0_str, dt_mid)))

    with T.time("chunk.extremum_loop"):
        _n_candidates = 0      # turning points below LEHMER_THR
        _n_verify = 0          # how many hit the expensive mp_Z_verify
        _n_precise = 0         # how many hit precise_gap
        _n_hits = 0
        # First, find all turning-point indices via pure array ops (fast). This
        # isolates the "find extrema" cost from the per-candidate work.
        with T.time("chunk.ext.find_turns"):
            Zc = Z[1:-1]; Zl = Z[:-2]; Zr = Z[2:]
            is_max_arr = (Zc > Zl) & (Zc > Zr)
            is_min_arr = (Zc < Zl) & (Zc < Zr)
            turn_mask = (is_max_arr | is_min_arr) & (np.abs(Zc) <= LEHMER_THR)
            turn_idx = np.nonzero(turn_mask)[0] + 1   # +1 to index back into Z
        for i in turn_idx:
            _n_candidates += 1
            is_max = Z[i] > Z[i-1] and Z[i] > Z[i+1]
            is_min = Z[i] < Z[i-1] and Z[i] < Z[i+1]
            with T.time("chunk.ext.parabola"):
                a = Z[i-1]; b = Z[i]; c = Z[i+1]
                denom = (a - 2*b + c)
                d = 0.5*(a - c)/denom if denom != 0 else 0.0
                dt_ext = dt_arr[i] + d*step
                z_vertex = b - 0.25*(a - c)*d
                Zpp = denom/(step*step)
                ngap_est = None
                if Zpp != 0 and (-2.0*z_vertex/Zpp) > 0:
                    ngap_est = (2.0*np.sqrt(-2.0*z_vertex/Zpp))/mg
            kind = 'max' if is_max else 'min'
            VERIFY_MARGIN = 1e-4
            wrong_side = (is_max and z_vertex < VERIFY_MARGIN) or \
                         (is_min and z_vertex > -VERIFY_MARGIN)
            if wrong_side:
                _n_verify += 1
                with T.time("chunk.ext.mp_Z_verify"):
                    z_true = mp_Z_verify(t0_str, dt_ext)
            else:
                z_true = z_vertex          # cheap float64 value is definitive
            with T.time("chunk.ext.exact_t_abs"):
                t_ext_abs = float(exact_t(t0_str, dt_ext))
            if is_max and z_true < 0:
                violations.append((t0_str, dt_ext, z_true, 'neg_local_max')); continue
            if is_min and z_true > 0:
                violations.append((t0_str, dt_ext, z_true, 'pos_local_min')); continue
            if abs(z_true) >= LEHMER_THR:
                continue
            gap = ngap = None
            if ngap_est is not None and ngap_est < NGAP_REFINE:
                _n_precise += 1
                with T.time("chunk.ext.precise_gap"):
                    g, lo, hi = precise_gap(t0_str, dt_ext, pool=pool, verbose=VALIDATE)
                if g is not None:
                    with T.time("chunk.ext.mean_gap_hit"):
                        gap = g; ngap = g/mean_gap(t_ext_abs)
            _n_hits += 1
            hits.append((t_ext_abs, z_true, kind, gap, ngap))
        if LIVE:
            print(f"    [ext] candidates={_n_candidates} verify={_n_verify} "
                  f"precise_gap={_n_precise} hits={_n_hits}", flush=True)

    return hits, violations, found, required, refined, dt_b, nz_tb, dt_arr, Z, nz_ta


def ultra_fine_recover(t0_str, pending_state, pool, ultra_fine_step=None):
    """Re-run a pending chunk at ULTRA-FINE grid resolution (default:
    FINE_STEP / 10 = 0.0001, i.e. 10x denser than the normal fine resweep,
    100x denser than the coarse grid) to find zeros the fine resweep missed.

    This is the "real miss" recovery path: called when a chunk's shortfall
    fails to self-heal at the next chunk's boundary. Ultra-fine resweep is
    expensive (roughly 10x a normal chunk's GPU work, ~30-40 minutes at
    1e13 on the reference RTX 3070 Ti hardware), so it only fires when the
    seam-heal path has ruled out boundary noise.

    Returns (recovered_count, new_dt_zeros, new_Z_left, new_Z_right,
             new_dt_left, new_dt_right):
    the number of newly-found zeros (should equal pending_state['shortfall']
    on success), and arrays with their locations and straddling Z values,
    ready to append to the zeros CSV.

    If recovered_count < shortfall, the deficit is a REAL persistent miss
    (ultra-fine couldn't find them either) -- caller decides how to log it.
    """
    _fs = ultra_fine_step if ultra_fine_step is not None else FINE_STEP / 10.0
    dt_a = pending_state['dt_a']
    dt_b = pending_state['seam']

    print(f"[ULTRA-FINE] Re-sweeping chunk {pending_state['chunk']} at "
          f"step={_fs:g} to recover {pending_state['shortfall']} missing "
          f"zeros. Expected wall time: ~{int(30 * (0.001/_fs))} minutes...",
          flush=True)

    with T.time("ultrafine.build_grid"):
        dt_ultra = np.arange(dt_a, dt_b, _fs)
        if dt_ultra[-1] < dt_b:
            dt_ultra = np.append(dt_ultra, dt_b)

    with T.time("ultrafine.compute_Z"):
        Z_ultra = compute_Z_array(t0_str, dt_ultra, pool)

    # Find every sign change in the ultra-fine sweep. These are all the
    # zeros in [dt_a, dt_b], including the ones we already knew about.
    sc_mask = (Z_ultra[:-1] >= 0) != (Z_ultra[1:] >= 0)
    ultra_zero_idxs = np.nonzero(sc_mask)[0]
    n_ultra_found = len(ultra_zero_idxs)
    print(f"[ULTRA-FINE] Found {n_ultra_found} zeros in resweep "
          f"(pending chunk originally found "
          f"{n_ultra_found - pending_state['shortfall']}).", flush=True)

    # Linear-interp the ultra-fine zeros to precise dt
    dz = Z_ultra[ultra_zero_idxs + 1] - Z_ultra[ultra_zero_idxs]
    with np.errstate(divide='ignore', invalid='ignore'):
        dt_ultra_zeros = np.where(
            dz != 0.0,
            dt_ultra[ultra_zero_idxs]
              - Z_ultra[ultra_zero_idxs]
              * (dt_ultra[ultra_zero_idxs + 1] - dt_ultra[ultra_zero_idxs]) / dz,
            0.5 * (dt_ultra[ultra_zero_idxs] + dt_ultra[ultra_zero_idxs + 1])
        )

    # Compare to previously-known zeros. Any ultra-fine zero that is not
    # within STEP of any previously-known zero is NEW (a recovered miss).
    # We use STEP (not FINE_STEP) as the match threshold because that's
    # the resolution the coarse sweep already handled -- an ultra-fine
    # zero within STEP of a known one is the same zero, just relocated
    # to higher precision.
    prev_zeros = np.asarray(pending_state['prev_zeros_dt'])
    if len(prev_zeros) > 0:
        # For each ultra zero, find nearest previous zero via searchsorted
        insert_pos = np.searchsorted(prev_zeros, dt_ultra_zeros)
        # Distance to nearest previous zero on either side
        left_pos = np.clip(insert_pos - 1, 0, len(prev_zeros) - 1)
        right_pos = np.clip(insert_pos, 0, len(prev_zeros) - 1)
        dist_left = np.abs(dt_ultra_zeros - prev_zeros[left_pos])
        dist_right = np.abs(dt_ultra_zeros - prev_zeros[right_pos])
        min_dist = np.minimum(dist_left, dist_right)
        new_mask = min_dist > STEP
    else:
        # No previous zeros known -- all ultra-fine zeros are "new"
        new_mask = np.ones(len(dt_ultra_zeros), dtype=bool)

    new_indices = np.nonzero(new_mask)[0]
    n_recovered = len(new_indices)
    print(f"[ULTRA-FINE] {n_recovered} zeros are new (not within STEP={STEP} "
          f"of any previously-known chunk zero). Expected: "
          f"{pending_state['shortfall']}.", flush=True)

    # Package up the new zeros for the caller to append to the CSV
    new_dt_zeros = dt_ultra_zeros[new_indices]
    new_Z_left   = Z_ultra[ultra_zero_idxs[new_indices]]
    new_Z_right  = Z_ultra[ultra_zero_idxs[new_indices] + 1]
    new_dt_left  = dt_ultra[ultra_zero_idxs[new_indices]]
    new_dt_right = dt_ultra[ultra_zero_idxs[new_indices] + 1]

    return (n_recovered, new_dt_zeros, new_Z_left, new_Z_right,
            new_dt_left, new_dt_right)


def append_recovered_zeros(t0_str, dt_zeros, Z_left, Z_right,
                            dt_left, dt_right, chunk_idx, baseline_indices):
    """Append recovered-from-ultra-fine zeros to the zeros CSV, with correct
    ordinal indices. Uses `-1000000 - chunk_idx` as the chunk marker, so
    recovered rows are unambiguously identifiable by `chunk < 0` (works even
    for chunk 0, which -0 would not distinguish from a normal chunk 0 row).
    To recover the original chunk from the marker: orig = -chunk - 1_000_000.

    baseline_indices: list of int, one per zero, giving the ORDINAL each
    recovered zero should have. Caller computes these by inserting the
    new zero's t position into the existing ordinal sequence."""
    if len(dt_zeros) == 0:
        return
    t0_mpf = mpf(t0_str)
    marker = -1_000_000 - abs(chunk_idx)
    with open(ZEROS_LOG, 'a') as f:
        for k in range(len(dt_zeros)):
            t_abs = float(t0_mpf + mpf(repr(float(dt_zeros[k]))))
            zi = baseline_indices[k]
            f.write(f"{t_abs:.10f},{Z_left[k]:+.6e},{Z_right[k]:+.6e},"
                    f"{dt_left[k]:.6f},{dt_right[k]:.6f},"
                    f"{marker},{zi}\n")


def bump_ordinals_after_recovery(t0_str, new_zero_t_abs_list, tz, tg):
    """After K new zeros have been recovered and appended to the CSV, every
    PREVIOUSLY-WRITTEN zero with t > any_new_zero_t needs its zero_index
    bumped up. Each existing zero's bump = (number of new zeros with t <
    existing zero's t). This restores strict ordinal correctness: after this
    call, `zero_index` monotonically increases with t across the entire CSV.

    Also updates leaderboard entries (tz, tg) in place: any entry with t >
    a new_zero_t gets its zero_index bumped by the same amount.

    Implementation is atomic: writes to a temp file first, then renames over
    the original. If interrupted mid-write, the original is untouched. On
    Windows the rename is os.replace which is atomic (unlike os.rename).

    Called only after successful ultra-fine recovery, so it fires at most
    a handful of times per long run. Cost: one full CSV pass -- for 1M zeros
    that's a few seconds. Acceptable given how rarely this runs.
    """
    if not new_zero_t_abs_list:
        return
    new_t = sorted(float(t) for t in new_zero_t_abs_list)

    def bump_for(t_val):
        """How many new zeros have t < t_val? That's how much this row's
        ordinal should be bumped up."""
        # binary search: find insertion point of t_val in new_t; that's the
        # count of new zeros with strictly smaller t.
        lo, hi = 0, len(new_t)
        while lo < hi:
            mid = (lo + hi) // 2
            if new_t[mid] < t_val:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # Rewrite the zeros CSV
    tmp_path = ZEROS_LOG + ".ordbump.tmp"
    n_bumped = 0
    # Precompute the exact t values of the just-inserted new zeros so we can
    # skip them (they already have correct ordinals). We use a set of the
    # formatted string forms as they appear in the CSV, so comparisons are
    # exact and not subject to float roundoff.
    new_t_strs = set(f"{t:.10f}" for t in new_t)
    with open(ZEROS_LOG, 'r') as fin, open(tmp_path, 'w') as fout:
        header = fin.readline()
        fout.write(header)
        cols = header.strip().split(',')
        try:
            t_ix = cols.index('t')
            zi_ix = cols.index('zero_index')
        except ValueError:
            # Old-format CSV without zero_index -- nothing to bump
            os.remove(tmp_path)
            return
        for line in fin:
            parts = line.rstrip('\n').split(',')
            if len(parts) <= zi_ix or not parts[zi_ix]:
                fout.write(line)
                continue
            try:
                t_val = float(parts[t_ix])
                zi_val = int(parts[zi_ix])
            except (ValueError, IndexError):
                fout.write(line)
                continue
            # Skip the newly-inserted recovered rows themselves (they already
            # have correct ordinals). Detect by exact t-string match.
            if parts[t_ix] in new_t_strs:
                fout.write(line)
                continue
            bump = bump_for(t_val)
            if bump > 0:
                parts[zi_ix] = str(zi_val + bump)
                fout.write(','.join(parts) + '\n')
                n_bumped += 1
            else:
                fout.write(line)
    # Atomic swap
    os.replace(tmp_path, ZEROS_LOG)

    # Also bump leaderboard entries in place. Each entry is a 6-tuple
    # (t, z, kind, gap, ngap, zero_index).
    for board in (tz, tg):
        for entry in board:
            if len(entry) >= 6 and entry[5] is not None:
                bump = bump_for(float(entry[0]))
                if bump > 0:
                    entry[5] = int(entry[5]) + bump

    print(f"[BUMP] Bumped {n_bumped:,} existing zero ordinals to account for "
          f"{len(new_t)} recovered zero(s) inserted mid-catalog.", flush=True)

# ---------------- checkpoint io ----------------
def load_ckpt():
    if not os.path.exists(CHECKPOINT):
        return {'next_chunk':0,'tightest_z':[],'tightest_gap':[],
                'violations':[], 'zeros_located':0,
                'zeros_required':0, 'zeros_short':0}
    with open(CHECKPOINT) as f: raw = json.load(f)
    nc = raw.get('next_chunk',0); vio = raw.get('violations',[])
    if 'tightest_z' in raw or 'tightest_gap' in raw:
        return {'next_chunk':nc,'tightest_z':raw.get('tightest_z',[]),
                'tightest_gap':raw.get('tightest_gap',[]),'violations':vio,
                'zeros_located':raw.get('zeros_located',0),
                'zeros_required':raw.get('zeros_required',0),
                'zeros_short':raw.get('zeros_short',0)}
    tz=[[e[0],e[1],e[2],None,None] for e in raw.get('tightest',[]) if len(e)>=3]
    return {'next_chunk':nc,'tightest_z':tz,'tightest_gap':[],
            'violations':vio,'zeros_located':0,'zeros_required':0,'zeros_short':0}

def save_ckpt(s):
    with open(CHECKPOINT,'w') as f: json.dump(s,f,indent=2)

def append_hits(hits):
    new = not os.path.exists(RESULTS_LOG)
    with open(RESULTS_LOG,'a') as f:
        if new: f.write("t,Z_mpmath,kind,gap,norm_gap,zero_index\n")
        for h in hits:
            # Accept both new 6-tuples and legacy 5-tuples so this doesn't
            # break if some path forgets to enrich (or if resume replays
            # old-format entries)
            if len(h) >= 6:
                t, z, kind, gap, ngap, zi = h[0], h[1], h[2], h[3], h[4], h[5]
            else:
                t, z, kind, gap, ngap = h; zi = None
            g  = f"{gap:.8e}" if gap is not None else ""
            ng = f"{ngap:.6f}" if ngap is not None else ""
            zs = str(zi) if zi is not None else ""
            f.write(f"{t:.10f},{z:.12e},{kind},{g},{ng},{zs}\n")


def append_zeros(t0_str, dt_arr, Z, chunk_idx, baseline_index=None):
    """
    Extract and log every zero located in this chunk.

    A sign change between consecutive samples Z[i], Z[i+1] contains exactly one
    zero (assuming the STEP is fine enough to resolve close pairs, which is
    guaranteed by the completeness check upstream -- if it wasn't, we already
    ran the fine resweep and this Z array is the refined one). Linear
    interpolation between the two samples gives the zero location to about
    STEP^2 / mean_gap precision -- e.g. at 1e13 with STEP=0.01 and mean_gap
    ~0.09, that's ~1e-3 absolute in dt, giving ~1e-3 precision on the absolute
    t (float64 has ~1e-3 absolute precision at 1e13 anyway, so this saturates
    the storage format cleanly).

    Every zero at t > 3e12 is BEYOND the continuously-verified frontier, so
    this catalog is the whole point of the run: locations that don't exist
    anywhere else. Also record the straddling Z values so tight pairs are
    identifiable after the fact (small |Z_left| and |Z_right| together = a
    pair the sweep resolved with margin; their signs and magnitudes let you
    reconstruct which zeros were the Lehmer-like near-misses).

    `baseline_index` is the ordinal index of the LAST zero BEFORE this chunk's
    first zero -- i.e. nzeros(t_a) where t_a is the chunk's safe_boundary_a.
    The scanner already computes this as `nz_ta` per chunk (for the
    completeness check), so passing it here is essentially free. The first
    zero in the chunk gets ordinal (baseline_index + 1), the second
    (baseline_index + 2), and so on. This makes every row's `zero_index`
    the TRUE ordinal position in the Riemann zeta zero sequence -- the
    field's standard way of referring to a zero. If baseline_index is None
    (backward-compatible with older callers), the zero_index column is
    written as an empty string.

    Returns the number of zeros written, so the caller can accumulate the
    running baseline for the next chunk (or verify it against nz_seam).
    """
    # find sign-change indices (vectorized, fast)
    sc = (Z[:-1] >= 0) != (Z[1:] >= 0)
    idx = np.nonzero(sc)[0]
    if len(idx) == 0:
        return 0
    # linear-interp zero locations in dt-space (small values, exact math)
    dz = Z[idx+1] - Z[idx]
    # guard against exact-zero denominator (Z[i]==Z[i+1]==0 is possible but
    # vanishingly rare; take midpoint in that pathological case)
    with np.errstate(divide='ignore', invalid='ignore'):
        dt_zero = np.where(
            dz != 0.0,
            dt_arr[idx] - Z[idx] * (dt_arr[idx+1] - dt_arr[idx]) / dz,
            0.5*(dt_arr[idx] + dt_arr[idx+1])
        )
    # reconstruct absolute t = t0 + dt at full precision, then cast to float64
    # for storage. t0 is a full-precision string; add small dt via mpf then
    # convert. Doing this once per zero is cheap (~10 microseconds each) and
    # gives the correct float64 value for a t that would otherwise lose
    # precision under naive float(t0)+dt at heights >= 1e13.
    t0_mpf = mpf(t0_str)
    new = not os.path.exists(ZEROS_LOG)
    with open(ZEROS_LOG, 'a') as f:
        if new:
            f.write("t,Z_left,Z_right,dt_left,dt_right,chunk,zero_index\n")
        for k, i in enumerate(idx):
            t_abs = float(t0_mpf + mpf(repr(float(dt_zero[k]))))
            if baseline_index is not None:
                zi = str(baseline_index + 1 + k)
            else:
                zi = ""
            f.write(f"{t_abs:.10f},{Z[i]:+.6e},{Z[i+1]:+.6e},"
                    f"{dt_arr[i]:.6f},{dt_arr[i+1]:.6f},{chunk_idx},{zi}\n")
    return len(idx)

# ---------------- main ----------------
if __name__ == '__main__':
    # ---- CLI (all flags optional; default behavior unchanged) ----
    _parser = argparse.ArgumentParser(add_help=True,
        description="Riemann-Siegel Z scanner (v6-HP). Flags are optional; "
                    "with no flags the script runs its module-level config.")
    _parser.add_argument("--json-progress", action="store_true",
        help="also emit structured JSON status events (tagged with "
             "'@@STATUS@@ ') for UI/coordinator consumption.")
    _parser.add_argument("--pause-flag", type=str, default=None, metavar="PATH",
        help="between chunks, exit cleanly if PATH exists (checkpoint safe).")

    # ---- Config overrides (compose freely; any subset OK) ----
    # These let a UI or a distributed coordinator drive the scanner without
    # editing source. Any not passed keep the module default. Together they
    # give full control over 'which chunks, at what height, with what step,
    # writing where' -- so 'do chunks 200-300 at 1e13 into results_200_300_*'
    # is one command line.
    _parser.add_argument("--t-base", type=float, default=None, metavar="VAL",
        help="override module T_BASE (starting height on the critical line). "
             "Accepts scientific notation e.g. 1e13.")
    _parser.add_argument("--chunk-t", type=float, default=None, metavar="VAL",
        help="override module CHUNK_T (dt units per chunk).")
    _parser.add_argument("--n-chunks", type=int, default=None, metavar="N",
        help="override module N_CHUNKS (upper bound of chunk loop). Combined "
             "with --start-chunk gives you a chunk range: e.g. --start-chunk "
             "200 --n-chunks 300 does chunks 200 through 299.")
    _parser.add_argument("--start-chunk", type=int, default=None, metavar="N",
        help="force starting chunk index, IGNORING any resume checkpoint. "
             "Useful for distributed work where a coordinator assigns each "
             "contributor a specific chunk range. Without this flag the "
             "scanner resumes from checkpoint['next_chunk'] as usual.")
    _parser.add_argument("--output-prefix", type=str, default=None, metavar="PREFIX",
        help="override checkpoint and CSV filenames all at once. Files become "
             "{PREFIX}_checkpoint.json, {PREFIX}_hits.csv, {PREFIX}_count.csv, "
             "{PREFIX}_zeros.csv. Use this to keep a distributed worker's "
             "output separate from any other run in the same directory.")

    _parser.add_argument("--work-unit", type=str, nargs=4, default=None,
        metavar=("T0","DT_START","DT_END","PREFIX"),
        help="[legacy] single-chunk atomic work unit. Equivalent to "
             "--t-base T0 --chunk-t (DT_END-DT_START) --start-chunk "
             "(DT_START/chunk_t) --n-chunks (start+1) --output-prefix PREFIX. "
             "Prefer the individual flags for new code.")
    _args = _parser.parse_args()
    _JSON_PROGRESS = _args.json_progress
    _PAUSE_FLAG_PATH = _args.pause_flag

    # ---- Apply config overrides in dependency order ----
    # T_BASE and CHUNK_T are needed to interpret --start-chunk (they define
    # what chunk N means in absolute t), so apply them first.
    if _args.t_base is not None:
        T_BASE = _args.t_base
    if _args.chunk_t is not None:
        CHUNK_T = _args.chunk_t
    if _args.n_chunks is not None:
        N_CHUNKS = _args.n_chunks
    _FORCE_START_CHUNK = _args.start_chunk    # None = resume from checkpoint
    if _args.output_prefix is not None:
        _pref = _args.output_prefix
        CHECKPOINT  = f"{_pref}_checkpoint.json"
        RESULTS_LOG = f"{_pref}_hits.csv"
        COUNT_LOG   = f"{_pref}_count.csv"
        ZEROS_LOG   = f"{_pref}_zeros.csv"

    # --work-unit is now a legacy convenience: translates to the flag set above
    _WU_DT_START = None
    if _args.work_unit is not None:
        _wu_t0, _wu_dts, _wu_dte, _wu_pref = _args.work_unit
        T_BASE = float(_wu_t0)
        _WU_DT_START = float(_wu_dts)
        _WU_DT_END = float(_wu_dte)
        CHUNK_T = _WU_DT_END - _WU_DT_START
        N_CHUNKS = 1
        _FORCE_START_CHUNK = 0
        CHECKPOINT  = f"{_wu_pref}_checkpoint.json"
        RESULTS_LOG = f"{_wu_pref}_hits.csv"
        COUNT_LOG   = f"{_wu_pref}_count.csv"
        ZEROS_LOG   = f"{_wu_pref}_zeros.csv"

    if not _check_torch(): raise SystemExit("CUDA GPU required.")
    import torch
    nproc = N_WORKERS or cpu_count()
    print("="*64)
    print(f"GPU scanner v6 [TIMED] - {torch.cuda.get_device_name(0)} + {nproc} cores")
    print("Parabolic gap + mpmath truth + Turing zero-count completeness")
    print("="*64)

    with T.time("startup.load_ckpt"):
        st = load_ckpt()
    sc = st['next_chunk']
    tz=st['tightest_z']; tg=st['tightest_gap']; av=st['violations']
    zver=st.get('zeros_located',0); zreq=st.get('zeros_required',0)
    short=st.get('zeros_short',0)
    prev_nz=st.get('prev_nz'); _restore_seam=st.get('prev_seam')
    # Cross-chunk boundary noise: rarely, mpmath.nzeros can be off by 1-2 at
    # the exact boundary point where safe_boundary lands (because Turing's
    # method has its own numerical precision at points where a zero sits very
    # close to t). This manifests as chunk N reporting "short by K" while
    # chunk N+1 reports "over by K" -- the totals net to zero across the
    # seam. We detect this: if chunk N is short, we hold the shortfall
    # pending; if chunk N+1 overshoots, we consume from pending (self-heal
    # and log it). If chunk N+1 does NOT overshoot, the pending is a REAL
    # miss and we auto-recover by re-running chunk N with an ULTRA-FINE
    # grid (FINE_STEP / 10 = 0.0001, so 10x denser than the standard fine
    # resweep). Any newly-found zeros are appended to the zeros CSV with
    # their correct ordinal indices. If ultra-fine STILL doesn't find them
    # (very unlikely), the shortfall counts as a real unresolved miss.
    #
    # Only the RESOLVED shortfall counts against `short` (the persistent
    # counter shown in status). pending_state carries enough context to
    # re-run the pending chunk if needed; None means no pending. Not
    # persisted across resume for now (worst case: one seam right at a
    # resume boundary needs manual re-check).
    pending_state = None    # or dict: {chunk, shortfall, dt_start, dt_end,
                            # dt_a, nz_ta, seam, nz_seam, prev_zeros_t}

    # --start-chunk overrides the checkpoint's resume position. Useful for
    # distributed work where a coordinator assigns explicit chunk ranges,
    # and the "checkpoint" this worker sees may be either empty (fresh output
    # prefix) or from an unrelated run. When overridden, we do NOT trust the
    # checkpoint's prev_seam/prev_nz either -- those are for the previous
    # chunk in the checkpoint's numbering, which is not our previous chunk.
    if _FORCE_START_CHUNK is not None:
        sc = _FORCE_START_CHUNK
        prev_nz = None            # force a fresh nzeros_a boundary call
        _restore_seam = None      # force a fresh safe_boundary_a call

    print(f"Chunk range: {sc}..{N_CHUNKS-1} (T_BASE={T_BASE:g}, "
          f"CHUNK_T={CHUNK_T:g}, first t={T_BASE+sc*CHUNK_T:.1f})")
    print(f"Zeros located by sweep so far: {zver}")
    if tg: print(f"Tightest true norm-gap: {tg[0][4]:.5f} @ t={tg[0][0]:.2f}")
    print()
    mpw = MP_WORKERS or cpu_count()
    t_run0=datetime.now()
    _recent_walls = []   # rolling last-10-chunk wall times for rate_recent
    with T.time("startup.pool_init"):
        pool=Pool(processes=mpw, initializer=_worker_init)
        # Dedicated tiny pool for async nzeros() calls -- lets nzeros_a and
        # nzeros_b run concurrently with the GPU compute_Z_array, saving
        # roughly the smaller of the two wall-times per chunk. 2 workers is
        # enough (one per boundary; chunks 2+ only use one). Kept separate
        # from the main 'pool' so nzeros doesn't fight the extremum-loop
        # parallel batching later.
        nz_pool=Pool(processes=2, initializer=_worker_init)
    prev_seam=_restore_seam if 'prev_nz' in st else None

    # Emit run_start with the run's config so a UI can populate its display.
    # Include resume-time cumulative counts and current leaderboard #1 so the
    # UI has meaningful state to show BEFORE the first chunk of this session
    # completes -- otherwise everything reads '—' during the first chunk,
    # which at 1e13 can be 6+ minutes.
    _resume_tightest = None
    if tg:
        # Support both new 6-wide entries and legacy 5-wide entries from
        # pre-index checkpoints. Old entries have zero_index = None.
        _row = tg[0]
        _t, _z, _kind, _gap, _ngap = _row[0], _row[1], _row[2], _row[3], _row[4]
        _zi = _row[5] if len(_row) >= 6 else None
        _resume_tightest = {"t": _t, "z": _z, "kind": _kind,
                            "gap": _gap, "norm_gap": _ngap,
                            "zero_index": _zi}
    _emit_status("run_start",
        t_base=T_BASE, chunk_t=CHUNK_T, n_chunks=N_CHUNKS, step=STEP,
        start_chunk=sc, checkpoint=CHECKPOINT,
        zeros_log=ZEROS_LOG, hits_log=RESULTS_LOG,
        zeros_located_resume=zver,
        zeros_required_resume=zreq,
        zeros_short_resume=short,
        violations_resume=len(av),
        tightest_resume=_resume_tightest,
        work_unit=(_args.work_unit is not None))

    try:
        t0_str = mp.nstr(mpf(T_BASE), 30)
        for c in range(sc, N_CHUNKS):
            # ---- Cooperative pause between chunks (no partial-chunk loss).
            # The current chunk's checkpoint was saved at the end of the
            # previous iteration; exiting here loses nothing.
            if _check_pause_flag():
                _emit_status("paused", next_chunk=c,
                    reason="pause flag file present")
                print(f"\n[pause] flag detected -- clean exit at chunk {c}.")
                break

            T.reset_chunk()          # start-of-chunk: clear per-chunk timers

            # --work-unit mode: use the caller-supplied dt range for this
            # single chunk; otherwise use the standard c*CHUNK_T layout.
            if _WU_DT_START is not None:
                dt_start = _WU_DT_START
                dt_end   = _WU_DT_END
            else:
                dt_start = c*CHUNK_T
                dt_end   = (c+1)*CHUNK_T
            t_start  = T_BASE + dt_start

            _emit_status("chunk_start", chunk=c, t_start=t_start,
                dt_start=dt_start, dt_end=dt_end)
            wall0=time.time()

            with T.time("main.process_chunk"):
                hits,viols,found,required,refined,seam,nz_seam,dt_final,Z_final,nz_ta_used = process_chunk(
                    t0_str, dt_start, dt_end, STEP, pool,
                    dt_a=prev_seam, nz_ta=prev_nz, nz_pool=nz_pool)

            prev_seam = seam
            prev_nz = nz_seam
            zver += found
            zreq += required

            # Cross-chunk boundary noise handling (see pending_state comment
            # near the top of __main__). Compute delta for THIS chunk, then:
            #   - if pending from prev chunk exists and this chunk overshoots,
            #     absorb the pending (self-heal): don't count it as a real short
            #   - if pending exists but this chunk didn't overshoot, escalate
            #     to ULTRA-FINE resweep on the pending chunk (auto-recovery)
            #   - if this chunk itself is short, set pending for next chunk
            delta = int(found) - int(required)
            _self_healed = 0
            _real_miss = 0
            _recovered = 0
            if pending_state is not None:
                if delta > 0:
                    # Self-heal: chunk N+1 overshot enough to absorb chunk N's
                    # pending shortfall. Confirms seam boundary noise, no
                    # real miss.
                    absorbed = min(delta, pending_state['shortfall'])
                    pending_state['shortfall'] -= absorbed
                    _self_healed = absorbed
                    print(f"[SEAM] chunk {pending_state['chunk']} boundary "
                          f"noise self-healed at chunk {c} "
                          f"(chunk {c} had +{delta} extra zeros, absorbed "
                          f"{absorbed}). No real miss.")
                    if pending_state['shortfall'] == 0:
                        pending_state = None
                else:
                    # No self-heal -- escalate to ultra-fine resweep on the
                    # PENDING chunk. This is expensive (~30-40 min at 1e13
                    # for FINE_STEP/10), but rare. Auto-recovers real misses
                    # so the dataset stays complete without human
                    # intervention.
                    print(f"[REAL MISS] chunk {pending_state['chunk']} short "
                          f"by {pending_state['shortfall']} zeros, NOT "
                          f"recovered by chunk {c} (chunk {c} delta="
                          f"{delta:+d}). Escalating to ultra-fine resweep.",
                          flush=True)
                    with T.time("main.ultra_fine_recover"):
                        n_recovered, new_dt_zeros, new_Zl, new_Zr, \
                            new_dtl, new_dtr = ultra_fine_recover(
                                t0_str, pending_state, pool)
                    if n_recovered > 0:
                        # Assign each recovered zero its TRUE ordinal (baseline
                        # + 1 + number of previous zeros with smaller dt +
                        # number of already-inserted recovered zeros with
                        # smaller dt), then bump every existing zero's ordinal
                        # by (number of new zeros with smaller t) to keep the
                        # whole catalog strictly monotonic in t. This is the
                        # mathematically correct fix -- inserting K zeros in
                        # the middle of a chunk shifts every subsequent zero's
                        # ordinal up by K, no exceptions.
                        prev_dt = np.asarray(pending_state['prev_zeros_dt'])
                        baseline = pending_state['nz_ta']
                        sort_order = np.argsort(new_dt_zeros)
                        new_dt_sorted = new_dt_zeros[sort_order]
                        new_Zl_sorted = new_Zl[sort_order]
                        new_Zr_sorted = new_Zr[sort_order]
                        new_dtl_sorted = new_dtl[sort_order]
                        new_dtr_sorted = new_dtr[sort_order]
                        new_ordinals = []
                        # Also collect the absolute t values of the new zeros
                        # for the ordinal bump pass below.
                        t0_mpf_local = mpf(t0_str)
                        new_t_abs = []
                        for i, new_dt in enumerate(new_dt_sorted):
                            n_prev_before = int(np.searchsorted(prev_dt, new_dt))
                            true_ordinal = baseline + 1 + n_prev_before + i
                            new_ordinals.append(true_ordinal)
                            new_t_abs.append(float(t0_mpf_local
                                                   + mpf(repr(float(new_dt)))))
                        append_recovered_zeros(t0_str, new_dt_sorted,
                            new_Zl_sorted, new_Zr_sorted,
                            new_dtl_sorted, new_dtr_sorted,
                            pending_state['chunk'], new_ordinals)
                        # Now bump every existing zero's ordinal in the CSV
                        # AND in the leaderboards, so the whole catalog stays
                        # monotonic in t.
                        with T.time("main.bump_ordinals"):
                            bump_ordinals_after_recovery(t0_str, new_t_abs,
                                                          tz, tg)
                        zver += n_recovered
                        _recovered = n_recovered
                        # Update the "short" counter with whatever the
                        # ultra-fine DIDN'T find (should be 0 in the normal
                        # case; positive if ultra-fine also came up short).
                        residual = pending_state['shortfall'] - n_recovered
                        if residual > 0:
                            short += residual
                            _real_miss = residual
                            print(f"[REAL MISS PERSISTENT] chunk "
                                  f"{pending_state['chunk']}: ultra-fine "
                                  f"recovered {n_recovered}/{pending_state['shortfall']} "
                                  f"missing zeros. {residual} still not "
                                  f"accounted for. These are real misses.")
                        else:
                            print(f"[RECOVERED] chunk "
                                  f"{pending_state['chunk']}: ultra-fine "
                                  f"successfully recovered all "
                                  f"{n_recovered} missing zeros. "
                                  f"Dataset now complete for that chunk.")
                    else:
                        # Ultra-fine came up empty on new zeros -- surprising.
                        # Count as real miss for accounting purposes.
                        short += pending_state['shortfall']
                        _real_miss = pending_state['shortfall']
                        print(f"[REAL MISS PERSISTENT] chunk "
                              f"{pending_state['chunk']}: ultra-fine resweep "
                              f"found 0 new zeros. Shortfall of "
                              f"{pending_state['shortfall']} counted as "
                              f"unresolved.")
                    pending_state = None
            if delta < 0:
                # This chunk was short. Compute this chunk's zero dt values
                # from Z_final so ultra-fine (if it fires) can identify which
                # zeros are NEW vs. which we already have. Also snapshot the
                # boundaries so a re-run doesn't need fresh safe_boundary or
                # nzeros calls.
                sc_mask_this = (Z_final[:-1] >= 0) != (Z_final[1:] >= 0)
                sc_idx_this = np.nonzero(sc_mask_this)[0]
                # linear-interp to get dt of each zero
                dz_this = Z_final[sc_idx_this + 1] - Z_final[sc_idx_this]
                with np.errstate(divide='ignore', invalid='ignore'):
                    dt_zeros_this = np.where(
                        dz_this != 0.0,
                        dt_final[sc_idx_this]
                          - Z_final[sc_idx_this]
                          * (dt_final[sc_idx_this + 1]
                             - dt_final[sc_idx_this]) / dz_this,
                        0.5 * (dt_final[sc_idx_this]
                               + dt_final[sc_idx_this + 1])
                    )
                pending_state = {
                    'chunk': c,
                    'shortfall': -delta,
                    'dt_start': dt_start,
                    'dt_end': dt_end,
                    'dt_a': (prev_seam if prev_seam is not None
                             else dt_final[0]),   # the chunk's true left edge
                    'seam': seam,                  # the chunk's true right edge
                    'nz_ta': nz_ta_used,
                    'nz_seam': nz_seam,
                    'prev_zeros_dt': dt_zeros_this,
                }
                # Note: we do NOT increment `short` here yet. It only gets
                # counted if the next chunk fails to absorb it AND ultra-fine
                # recovery also fails to find the missing zeros.

            wall=time.time()-wall0

            # Log every zero located in this chunk. Every t > 3e12 is past the
            # continuously-verified frontier, so this catalog IS the science.
            # baseline_index = nzeros(t_a) tells append_zeros the ordinal of the
            # last zero BEFORE this chunk's first zero; each row gets its true
            # index in the Riemann zeta zero sequence (baseline + 1, +2, ...).
            append_zeros(t0_str, dt_final, Z_final, c, baseline_index=nz_ta_used)

            # Enrich each hit with its zero_index. A near-miss extremum sits
            # BETWEEN two consecutive zeros; the "index" we record is the
            # ordinal of the LEFT zero of the pair (so its right neighbor is
            # zero_index+1). We compute this cheaply by counting sign changes
            # in Z_final that occur before the extremum's dt position.
            if hits and nz_ta_used is not None:
                # Positions of all sign changes in the (possibly refined) grid
                sc_mask = (Z_final[:-1] >= 0) != (Z_final[1:] >= 0)
                sc_dt = dt_final[:-1][sc_mask]   # dt of left-of-sign-change sample
                enriched = []
                for h in hits:
                    t_ext = h[0]
                    # h[0] is absolute t; convert to dt for this chunk
                    dt_ext = t_ext - float(mpf(t0_str))
                    n_before = int(np.searchsorted(sc_dt, dt_ext, side='right'))
                    # The extremum sits after `n_before` sign-changes-so-far,
                    # so the left zero of the bracketing pair has ordinal
                    # nz_ta + n_before (that's the count of zeros at or before
                    # the extremum, i.e. the ordinal of the last zero <= t_ext).
                    zi = nz_ta_used + n_before
                    enriched.append((*h, zi))
                hits = enriched
            elif hits:
                # baseline unknown -> pad with None so tuple width is consistent
                hits = [(*h, None) for h in hits]

            if hits:
                with T.time("main.append_hits"):
                    append_hits(hits)
                for h in hits:
                    tz.append(list(h))
                    if h[4] is not None: tg.append(list(h))
                tz.sort(key=lambda x: abs(x[1])); tz=tz[:50]
                tg.sort(key=lambda x: x[4] if x[4] is not None else 9.0); tg=tg[:50]
            if viols:
                with T.time("main.verify_violations"):
                    for vt0, vdt, z, kind in viols:
                        is_viol, dt_true, Z_true = verify_violation(vt0, vdt, kind, dps=100)
                        if not is_viol:
                            continue
                        is_viol2, dt2, Z2 = verify_violation(vt0, dt_true, kind, dps=200)
                        if not is_viol2:
                            continue
                        t2 = float(exact_t(vt0, dt2))
                        av.append([t2, Z2, kind])
                        _emit_status("violation_survived", chunk=c,
                            t=t2, Z=Z2, kind=kind)
                        print("\n"+"!"*64)
                        print(f"RH VIOLATION SURVIVED dps=200: {kind}")
                        print(f"  t={t2:.10f}  Z={Z2:+.12e}")
                        print(f"  (cleared noise margin; investigate by hand NOW)")
                        print("!"*64+"\n")

            st['next_chunk']=c+1; st['tightest_z']=tz; st['tightest_gap']=tg
            st['violations']=av; st['zeros_located']=zver
            st['zeros_required']=zreq; st['zeros_short']=short
            st['prev_nz']=prev_nz; st['prev_seam']=prev_seam
            with T.time("main.save_ckpt"):
                save_ckpt(st)
            with T.time("main.empty_cache"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # chunk_end: emit AFTER checkpoint save, so a UI reading this
            # event knows the on-disk state is consistent with the numbers
            # reported. Includes rate, cumulative counts, and current tightest.
            #
            # Three rates so both "how fast right now" and "how fast on
            # average" are visible:
            #   rate_inst   = CHUNK_T / wall            (this chunk alone)
            #   rate_recent = avg of last 10 chunks     (rolling, smooths spikes)
            #   rate_cum    = units_run / elapsed       (whole session avg)
            # rate_inst is what matches the wall time the user is watching.
            # rate_cum was the only rate before, but it heavily lags after any
            # optimization lands because early-session slow chunks dominate.
            rate_inst = CHUNK_T / max(wall, 1e-9)
            _recent_walls.append(wall)
            if len(_recent_walls) > 10:
                _recent_walls.pop(0)
            rate_recent = (CHUNK_T * len(_recent_walls)) / max(sum(_recent_walls), 1e-9)
            units_run = (c+1-sc)*CHUNK_T
            rate_cum = units_run / max((datetime.now()-t_run0).total_seconds(), 1e-9)

            _tightest = None
            if tg:
                # tg[0] is a 6-tuple/list (t, z, kind, gap, ngap, zero_index)
                # for entries created after this update; older resumed
                # entries may be 5-wide, so guard the trailing field.
                _zi = tg[0][5] if len(tg[0]) >= 6 else None
                _tightest = {"t": tg[0][0], "z": tg[0][1], "kind": tg[0][2],
                             "gap": tg[0][3], "norm_gap": tg[0][4],
                             "zero_index": _zi}
            _emit_status("chunk_end", chunk=c, wall=wall,
                found=int(found), required=int(required),
                short=int(max(0, required-found)),
                hits=len(hits), refined=bool(refined),
                zeros_located_total=int(zver),
                zeros_required_total=int(zreq),
                zeros_short_total=int(short),
                # Cross-chunk seam accounting: how much of this chunk's
                # overshoot was consumed by a prior chunk's pending short
                # (self-heal), how much was auto-recovered by ultra-fine
                # resweep, how much of the pending was declared a real miss
                # (ultra-fine also came up short), and the current pending
                # state. `pending_short` here is what's currently unresolved
                # awaiting the NEXT chunk's decision.
                self_healed=int(_self_healed),
                recovered=int(_recovered),
                real_miss=int(_real_miss),
                pending_short=int(pending_state['shortfall']
                                  if pending_state is not None else 0),
                pending_short_chunk=(pending_state['chunk']
                    if pending_state is not None else -1),
                rate_t_per_s=rate_inst,        # keep field name for UI compat;
                                               # now means INSTANTANEOUS
                rate_recent_t_per_s=rate_recent,
                rate_cumulative_t_per_s=rate_cum,
                tightest=_tightest)

            if c % 5 == 0 or hits or viols or refined or found<required:
                gstr=(f"{tg[0][4]:.4f}@{tg[0][0]:.1f}" if tg else "none")
                ck = "OK" if found>=required else f"SHORT{found-required}"
                rf = "*" if refined else " "
                print(f"chunk {c:6d} t={t_start:.0f} {wall:5.1f}s{rf}"
                      f"hits={len(hits)} N={found}/{required}[{ck}] "
                      f"rate={rate_inst:.1f}t/s "
                      f"(10ch avg {rate_recent:.1f}, cum {rate_cum:.1f}) "
                      f"tightest={gstr}")
                # Per-chunk timing breakdown always printed on log lines,
                # so you can watch idle vs GPU vs pool.map live.
                print(T.format_chunk(wall))
    except KeyboardInterrupt:
        print("\nInterrupted - checkpoint saved.")
        _emit_status("interrupted", zeros_located_total=int(zver))
    finally:
        pool.close(); pool.join()
        nz_pool.close(); nz_pool.join()
        # Cumulative totals ALWAYS printed, including on Ctrl-C.
        print(T.format_totals())

    print(f"\nZeros located by sweep: {zver}")
    print(f"Zeros required (nzeros): {zreq}")

    # If the LAST chunk had a shortfall we're still holding as pending (no
    # subsequent chunk got to absorb it via overshoot), the honest thing is
    # to escalate to ultra-fine resweep directly rather than assume the seam
    # would have self-healed. This treats the end-of-run pending exactly the
    # same as a mid-run persistent shortfall.
    if pending_state is not None:
        print(f"[END-OF-RUN PENDING] Final chunk {pending_state['chunk']} had "
              f"shortfall {pending_state['shortfall']} with no following "
              f"chunk to check for seam self-heal. Escalating to ultra-fine "
              f"resweep directly.", flush=True)
        with T.time("main.ultra_fine_recover_endrun"):
            n_recovered, new_dt_zeros, new_Zl, new_Zr, \
                new_dtl, new_dtr = ultra_fine_recover(
                    t0_str, pending_state, pool)
        if n_recovered > 0:
            prev_dt = np.asarray(pending_state['prev_zeros_dt'])
            baseline = pending_state['nz_ta']
            sort_order = np.argsort(new_dt_zeros)
            new_dt_sorted = new_dt_zeros[sort_order]
            new_Zl_sorted = new_Zl[sort_order]
            new_Zr_sorted = new_Zr[sort_order]
            new_dtl_sorted = new_dtl[sort_order]
            new_dtr_sorted = new_dtr[sort_order]
            new_ordinals = []
            t0_mpf_local = mpf(t0_str)
            new_t_abs = []
            for i, new_dt in enumerate(new_dt_sorted):
                n_prev_before = int(np.searchsorted(prev_dt, new_dt))
                new_ordinals.append(baseline + 1 + n_prev_before + i)
                new_t_abs.append(float(t0_mpf_local
                                       + mpf(repr(float(new_dt)))))
            append_recovered_zeros(t0_str, new_dt_sorted, new_Zl_sorted,
                new_Zr_sorted, new_dtl_sorted, new_dtr_sorted,
                pending_state['chunk'], new_ordinals)
            # Bump every existing zero's ordinal to keep the catalog strictly
            # monotonic in t after these insertions.
            bump_ordinals_after_recovery(t0_str, new_t_abs, tz, tg)
            zver += n_recovered
            residual = pending_state['shortfall'] - n_recovered
            if residual > 0:
                short += residual
                print(f"[END-OF-RUN] chunk {pending_state['chunk']}: "
                      f"ultra-fine recovered {n_recovered}/"
                      f"{pending_state['shortfall']}; {residual} still "
                      f"unresolved.")
            else:
                print(f"[END-OF-RUN RECOVERED] chunk "
                      f"{pending_state['chunk']}: ultra-fine successfully "
                      f"recovered all {n_recovered} missing zeros.")
        else:
            short += pending_state['shortfall']
            print(f"[END-OF-RUN REAL MISS] chunk {pending_state['chunk']}: "
                  f"ultra-fine found 0 new zeros. Shortfall counted as "
                  f"unresolved.")
        pending_state = None

    print(f"Still short after refine: {short}  "
          f"({100*(zreq-short)/max(zreq,1):.4f}% complete)")
    if short==0:
        print("COMPLETE: every zero in the swept range located and on the line.")
    print("\n=== Tightest TRUE normalized zero-pair spacings ===")
    for e in tg[:15]:
        # 6-tuple (with zero_index) or legacy 5-tuple
        t, z, kind, gap, ngap = e[0], e[1], e[2], e[3], e[4]
        zi = e[5] if len(e) >= 6 else None
        zi_str = f"  #{zi:,}" if zi is not None else ""
        print(f"  t={t:.6f} norm_gap={ngap:.5f} gap={gap:.6e} "
              f"Z={z:+.3e} ({kind}){zi_str}")

    _emit_status("run_end",
        zeros_located_total=int(zver), zeros_required_total=int(zreq),
        zeros_short_total=int(short),
        complete=bool(short==0),
        top_tightest=[{"t": e[0], "z": e[1], "kind": e[2],
                       "gap": e[3], "norm_gap": e[4],
                       "zero_index": (e[5] if len(e) >= 6 else None)}
                      for e in tg[:15]])