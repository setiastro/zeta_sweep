# zeta_sweep

A GPU-accelerated Riemann–Siegel Z scanner that catalogs zeros of the
Riemann zeta function ζ(s) on the critical line, verifies completeness
against `mpmath.nzeros`, and measures Lehmer-type tight-pair spacings.

Built as a serious hobby project by [Franklin Marek](https://setiastro.com)
(SetiAstro). Distributes both the compute engine and a PyQt6 UI wrapper
for pause/resume-able runs. Designed with distributed work assignment in
mind, GIMPS-style.

## What this is (and, importantly, what it is not)

**What this is.** A tool that computes Z(t) on a dense grid across a
requested range of heights, locates every zero in that range, verifies
the located count matches `mpmath.nzeros(t_b) − mpmath.nzeros(t_a)` per
chunk, and records tight-pair spacings (Lehmer-phenomenon near-misses)
to high precision. Every chunk that reports `[OK]` means the completeness
count was checked and passed.

**What this is not.** This is not a claim about the Riemann Hypothesis
beyond the swept range. It does not attempt a proof or disproof. It
records that the zeros located in a specific interval `[t_min, t_max]`
lie on the critical line to the precision of the calculation, and that
their count matches the analytic zero-count formula in that interval.
That's a computational contribution, not a theoretical one.

**Rigor context.** The current published rigorous frontier of continuous
RH verification is **height ~3 × 10¹²**, from Platt & Trudgian ([*The
Riemann hypothesis is true up to 3·10¹²*](https://arxiv.org/abs/2004.09765),
*Bull. Lond. Math. Soc.* 53 (2021) 792–797), which used interval
arithmetic on university HPC resources. This tool works past that
frontier at **height ~10¹³** (published dataset covers t ∈ [10¹³,
10¹³ + 10⁵]) using GPU-accelerated Riemann–Siegel with mpmath ground
truth for the remainder term and for `nzeros` completeness. It does not
use interval arithmetic — the precision guarantee is via independent
mpmath cross-verification of every extremum below the near-miss
threshold, plus completeness certification against `nzeros`. This is
less rigorous than Platt–Trudgian's interval-arithmetic method but is
still a defensible computational catalog.

## Published dataset (first release)

**Release:** [`zeta_1e13+100chunks`](https://github.com/setiastro/zeta_sweep/releases/tag/zeta_1e13%2B100chunks)

- **Range swept:** t ∈ [10¹³, 10¹³ + 100,000]
- **Zeros located:** 447,157
- **Zeros required (`nzeros`):** 447,157
- **Shortfall after refine:** 0 (100.0000% complete)
- **Violations found:** 0
- **Tightest normalized gap:** 0.01124 at t = 10,000,000,097,148.81
  (actual gap ≈ 2.51 × 10⁻³, extremum Z ≈ −7.0 × 10⁻⁵)

**Files in the release:**

| File                                | Description                                                                                                                      |
| :---------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| `zeta_zeros_v6_1p0e13.csv` (~36 MB) | Every zero located. Linear-interpolated location per zero, with straddling Z values for provenance. See schema below.            |
| `zeta_hits_v6_1p0e13.csv`           | Near-misses (extrema with \|Z\| < 0.02). Each has an mpmath-verified extremum value and (for the tightest) a precise-gap measurement. |
| `zeta_scan_v6_1p0e13.json`          | Checkpoint with cumulative counts, top-50 tightest-Z and top-50 tightest-gap leaderboards, resume state.                          |

The zeros CSV is in the release rather than the repo root because it's
36 MB — too large to comfortably ship as a git-tracked file. Later
releases will contain expanded sweeps as they complete.

## Method (short version)

1. **Chunk-based sweep.** The range is broken into chunks of `CHUNK_T`
   units (default 1000). Each chunk samples Z(t) on a grid of step
   `STEP` = 0.01.

2. **GPU-parallel main sum.** Z(t) = 2 · Σ cos(θ(t) − t·ln n) / √n
   for n = 1..N with N = √(t/2π). Computed for the whole chunk's grid
   in one CUDA operation (float64). At t = 10¹³, N ≈ 1.26 million per
   sample.

3. **High-precision argument reduction.** At height 10¹³ the cosine
   argument reaches ~10¹⁴ and native float64 quantizes past the sample
   step. The code uses a `(t₀, dt)` grid split — the anchor `t₀` is held
   at full mpmath precision, and only the small offset `dt` participates
   in float64 arithmetic. This keeps the argument reduction exact where
   it matters (see `exact_t` helper).

4. **Riemann–Siegel remainder from mpmath.** R(t) is computed by
   `mpmath.siegelz` at anchor points across the chunk, interpolated to
   grid points. Not derived from the GPU float64 result — an independent
   ground truth.

5. **Completeness check via `nzeros`.** After the coarse sweep counts
   sign changes, the code calls `mpmath.nzeros(t_b) − mpmath.nzeros(t_a)`
   for the chunk's t-range and compares. If the count is short, the
   chunk is re-swept at 10× denser resolution (`FINE_STEP` = 0.001) to
   catch tight pairs the coarse grid missed. This is what the `[OK]`
   marker on each chunk certifies.

6. **Extremum extraction.** Turning points with |Z| < LEHMER_THR = 0.02
   are candidates for near-misses. A parabolic fit through three grid
   samples estimates the extremum value and location. Wrong-side
   extrema (a positive local minimum or a negative local maximum)
   trigger an mpmath re-verification at 50-digit precision. If the
   wrong-side interpretation survives, it escalates to dps=100 and
   then dps=200 before being reported as a candidate RH violation. To
   date, no candidate has survived past dps=100 in the swept range.

7. **Precise gap measurement.** For each tight near-miss, the two zeros
   bracketing the extremum are located by parallel outward scan +
   parallel bisection using `mpmath.siegelz`, producing gap
   measurements accurate to ~10⁻¹¹.

## Validation ladder

The scanner has been re-validated at each height:

| Test point            | Notes                                                                                                                 |
| :-------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| **t ≈ 7005.082**      | Reproduces the classic Lehmer near-miss: `gap = 3.769850e-02`, `norm_gap = 0.04210`, extremum Z = +0.003967 (max). |
| **t = 2.5 × 10¹⁰**    | 100+ chunks, count matches `nzeros` exactly on every chunk.                                                          |
| **t = 10¹²**          | 100 chunks, 410,511 / 410,511 zeros, no shortfall.                                                                    |
| **t = 10¹³**          | 100 chunks, 447,157 / 447,157 zeros, no shortfall. This is the published dataset.                                    |

## Files in the repo

| File                       | Description                                                                                    |
| :------------------------- | :--------------------------------------------------------------------------------------------- |
| `zeta_gpu_scan_v6_hp.py`   | The scanner engine. Runs headless from CLI; also drives the UI via subprocess.                 |
| `zeta_scanner_ui.py`       | PyQt6 UI wrapper. Start / Pause / Resume / Abort, live status, output mirror.                  |
| `zeta_hits_v6_1p0e13.csv`  | Near-miss leaderboard from the first published sweep (small enough to ship in-repo).           |
| `zeta_scan_v6_1p0e13.json` | Checkpoint from the first published sweep, including the top-50 tightest pairs.                |
| `LICENSE`                  | MIT.                                                                                           |

The bulk zeros catalog for the first sweep is in the
[GitHub release](https://github.com/setiastro/zeta_sweep/releases/tag/zeta_1e13%2B100chunks),
not the repo tree, due to size.

## Requirements

- Python 3.10+
- CUDA-capable GPU (float64-capable; RTX 30-series or newer recommended)
- `pip install numpy torch mpmath PyQt6`

The first-sweep dataset was produced on: Xeon E5-2696v3 (18 cores / 36
threads), 256 GB RAM, RTX 3070 Ti (8 GB VRAM). A single 1000-unit chunk
at t = 10¹³ takes roughly 3–7 minutes; refined chunks (that hit the fine
resweep for a tight pair) can take 30+ minutes. Sustained rate around
1.4 units of t per second at 10¹³.

## Running headless

```
python zeta_gpu_scan_v6_hp.py
```

Runs the module-default config (edit `T_BASE`, `N_CHUNKS`, and filenames
at the top of the file). Outputs a checkpoint, a hits CSV, a zeros CSV,
and a count log to the current directory.

**Optional CLI flags** (all composable, all optional):

```
--t-base VAL            override module T_BASE (e.g. 1e13)
--chunk-t VAL           override CHUNK_T (dt units per chunk)
--n-chunks N            upper bound on chunk index
--start-chunk N         force starting chunk (ignore resume checkpoint)
--output-prefix PREFIX  rename output files: {PREFIX}_hits.csv, etc.
--json-progress         emit structured @@STATUS@@ JSON events for UI/coordinator
--pause-flag PATH       between chunks, exit cleanly if PATH exists
```

To run "chunks 200 through 299 at 10¹³" for a coordinator-style split:

```
python zeta_gpu_scan_v6_hp.py \
  --t-base 1e13 --start-chunk 200 --n-chunks 300 \
  --output-prefix chunks_200_300
```

That worker's four files (`chunks_200_300_zeros.csv`, `_hits.csv`,
`_count.csv`, `_checkpoint.json`) can then be merged with anyone else's
by concatenating the CSVs and deduping by t. The seam between adjacent
workers has a small overlap zone where zeros may appear twice pre-dedupe;
this is a known limitation of the current fresh-start-per-worker design.

## Running with the UI

```
python zeta_scanner_ui.py
```

A single window with:
- Configuration fields for scanner path, T_BASE, N_CHUNKS, start-chunk,
  output prefix, pause-flag path, and working directory
- State-aware **Start / Pause / Resume / Abort** buttons
- Progress bar with chunk N of M, elapsed / ETA
- Cumulative statistics: zeros located, required, short-after-refine, violations
- Tightest pair display (t, norm_gap, gap, Z, kind)
- Scrollable live-output pane mirroring the scanner's stdout

Pause is cooperative — the scanner finishes the current chunk, saves its
checkpoint, and exits cleanly. Resume relaunches and the existing
checkpoint picks up exactly where it stopped. Abort terminates the
subprocess; previously completed chunks remain safely checkpointed.

## Data schemas

### `zeta_zeros_*.csv` (bulk catalog)

```
t, Z_left, Z_right, dt_left, dt_right, chunk
```

- `t` — interpolated absolute zero location (float64 precision; at 10¹³
  the ULP is ~2 × 10⁻³, saturating the storage format)
- `Z_left`, `Z_right` — Z values at the two straddling grid samples;
  smaller magnitudes → more precise linear-interp, or a tight-pair
  neighborhood
- `dt_left`, `dt_right` — grid-sample offsets that bracket this zero.
  `dt_right − dt_left ≈ 0.01` for a normal coarse-grid zero;
  `≈ 0.001` for a zero recovered by the fine resweep (higher precision
  automatically — free provenance)
- `chunk` — chunk index within the run

### `zeta_hits_*.csv` (near-misses)

```
t, Z_mpmath, kind, gap, norm_gap
```

- `t` — location of the near-miss extremum (parabolic vertex, float64)
- `Z_mpmath` — extremum value evaluated at high precision (dps=50);
  small |Z| = axis-hugging extremum
- `kind` — `max` or `min`
- `gap` — precise measurement of the spacing between the two zeros
  bracketing this extremum, from parallel mpmath bisection (~10⁻¹¹)
- `norm_gap` — `gap` divided by local mean spacing (2π/ln(t/2π));
  small values are the interesting ones

### `zeta_scan_*.json` (checkpoint)

Cumulative counts, resume state, and both leaderboards (top-50 tightest
by |Z| at extremum, top-50 tightest by normalized gap).

## Loading the data

CSVs work with anything that isn't Excel. Excel caps at ~1M rows and
struggles with 36 MB files; use pandas / numpy / SQLite / awk instead:

```python
import pandas as pd
zeros = pd.read_csv("zeta_zeros_v6_1p0e13.csv")
print(f"{len(zeros):,} zeros in [{zeros.t.min():.0f}, {zeros.t.max():.0f}]")

hits = pd.read_csv("zeta_hits_v6_1p0e13.csv")
print(hits.nsmallest(10, "norm_gap"))
```

## Verifying a zero independently

Any row from the zeros CSV can be spot-checked in mpmath:

```python
from mpmath import mp, mpf, siegelz, nzeros
mp.dps = 50

t = mpf("10000000097148.814453")   # tightest-pair extremum from the leaderboard
print(f"Z(t) = {float(siegelz(t)):+.4e}")   # should be near-zero (~1e-4 magnitude)

# Confirm two zeros bracket this extremum:
t_a = mpf("10000000097148.80"); t_b = mpf("10000000097148.83")
print(f"zeros in [{t_a}, {t_b}]: {int(nzeros(t_b) - nzeros(t_a))}")
```

## Contributing chunks

If you want to help sweep more of the frontier: pick an unclaimed chunk
range at some agreed T_BASE (open an issue to coordinate), run the
scanner as above with `--start-chunk`, `--n-chunks`, and
`--output-prefix`, and send back the four output files. A merge script
for combining contributor outputs is a planned addition.

## Explicit non-claims

- Zeros outside the swept range are not addressed by this data.
- No claim is made about the truth of the Riemann Hypothesis beyond
  what the swept range and completeness check certify.
- The precision guarantee is via independent mpmath cross-verification
  and `nzeros` completeness, not interval arithmetic. For a stricter
  interval-arithmetic result up to ~3 × 10¹², see Platt & Trudgian
  (2020).
- The precise-gap measurements are to ~10⁻¹¹; the bulk-catalog zero
  locations are to float64 storage precision (~10⁻³ absolute at t = 10¹³,
  which is the storage limit rather than the algorithm's).

## License

MIT — see `LICENSE`. Data in the releases is also under MIT.

## Citation

If you use this data or code, a citation like:

> Marek, F. (SetiAstro), *zeta_sweep: GPU-accelerated Riemann–Siegel
> Z-function sweep with completeness certification.* GitHub, 2026.
> https://github.com/setiastro/zeta_sweep

is appreciated. Specific datasets should also cite the release tag
(e.g. `zeta_1e13+100chunks`) so the exact range and code version are
identifiable.

## Acknowledgments

Built on top of `mpmath`'s excellent implementations of `siegelz`,
`siegeltheta`, and `nzeros`. Method inspired by the classical
approach of Riemann–Siegel with modern GPU parallelism for the
main sum. Independent verification of the 2020 Platt–Trudgian frontier
was the motivating goal.
