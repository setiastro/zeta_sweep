# zeta_sweep

**A GPU-accelerated Riemann–Siegel Z scanner for cataloging zeros of the Riemann zeta function ζ(s) on the critical line, with end-to-end completeness certification.** Built as a hobby project by [Franklin Marek](https://setiastro.com) (SetiAstro). Distributes both the compute engine and a PyQt6 desktop UI wrapper for pause/resume-able runs. Designed with GIMPS-style distributed contributions in mind.

The first release ([`zeta_1e13+224kT`](https://github.com/setiastro/zeta_sweep/releases/tag/zeta_1e13%2B224kT)) catalogs **1,001,632 zeros** at height t ≈ 10¹³, spanning ordinals **#43,124,192,297,104 through #43,124,193,298,735**. Every zero is verified on the critical line; count matches `mpmath.nzeros` exactly with zero shortfall.

---

## What this is (for anyone)

The **Riemann zeta function** ζ(s) is one of the most important objects in mathematics. Its "non-trivial zeros" — the complex numbers where ζ(s) = 0, excluding a well-understood infinite family on the negative real axis — encode deep information about the distribution of prime numbers.

The **Riemann Hypothesis** (RH), formulated in 1859, conjectures that *every* non-trivial zero lies on a single vertical line in the complex plane called the "critical line" (the line where the real part equals 1/2). RH is one of the seven Clay Millennium Prize Problems, with a $1 million prize for a proof or disproof. It has been checked for the first several trillion zeros without exception — every one lies on the critical line — but a proof remains elusive.

**Cataloging zeros** is a way of contributing observational evidence toward (or, potentially, against) RH. Every additional zero we verify on the critical line strengthens the empirical case; conversely, a single zero found off the line would disprove RH outright. Beyond that, the *spacings* between zeros carry their own information: they behave statistically like eigenvalues of large random matrices (this is the [Montgomery pair-correlation conjecture](https://en.wikipedia.org/wiki/Montgomery%27s_pair_correlation_conjecture)), and unusually tight pairs of zeros (called [Lehmer phenomena](https://en.wikipedia.org/wiki/Lehmer_pair)) are of independent mathematical interest.

**What this project does**: computes and catalogs zeros of ζ(s) at heights past the current rigorously-verified frontier, using consumer GPU hardware. Each zero we locate is annotated with its true position in the Riemann zero sequence (its "ordinal"), so the data is joinable to any other zero catalog by integer key. The catalog is small compared to what large HPC installations could produce, but it's genuinely new data — nobody has continuously cataloged zeros in this specific t range before with ordinal labels.

## For mathematicians (the technical version)

**Explicit scope.** This dataset catalogs the non-trivial zeros of ζ(s) with imaginary parts in the interval t ∈ [10¹³, 10¹³ + 223,999.95], corresponding to ordinal indices n ∈ [43,124,192,297,104, 43,124,193,298,735]. Every zero in this interval is located and its imaginary part reported to float64 precision (≈2×10⁻³ absolute at this height, which saturates the storage format). Every zero is confirmed to lie on the critical line to the precision of the calculation (mpmath at dps=50, escalated to dps=100 or 200 on any suspected off-critical-line extremum). The count of located zeros matches `mpmath.nzeros(t_end) − mpmath.nzeros(t_start)` exactly (difference = 0).

**Relation to prior computational work.** The current rigorous continuous-verification frontier is due to Platt & Trudgian ([*The Riemann hypothesis is true up to 3·10¹²*](https://arxiv.org/abs/2004.09765), *Bull. Lond. Math. Soc.* 53 (2021) 792–797), who used interval arithmetic on university HPC resources to verify RH continuously up to height ~3 × 10¹² (the first 12,363,153,437,138 zeros). Odlyzko has computed spot samples of zeros at much greater heights (e.g., around 10²²), but not continuously. This dataset extends the *continuous* frontier upward to ~10¹³ + 224,000 for the specific interval covered. It does not extend the *rigorous* frontier — see the Reliability section for the epistemic difference.

**Precision claim.** Zero locations are reported to float64 precision (~10⁻³ absolute at height 10¹³, which is the storage-format limit; the algorithm's linear-interp precision on the coarse grid is ~STEP²/mean_gap ≈ 1×10⁻³ so we don't leave precision on the table). Precise-gap measurements for tight pairs are computed via parallel mpmath bisection at dps=50 and are accurate to ~10⁻¹¹. Ordinal indices are exact integers.

**Non-claims.** This is not a proof of RH. It does not use interval arithmetic and therefore is not rigorous in the Platt–Trudgian sense — the precision guarantee is via independent mpmath cross-verification and `nzeros` completeness certification, not certified enclosures. See the Reliability section below for full detail.

---

## Dataset at a glance

| Property | Value |
| :--- | :--- |
| **Release tag** | [`zeta_1e13+224kT`](https://github.com/setiastro/zeta_sweep/releases/tag/zeta_1e13%2B224kT) |
| **t range** | [10¹³, 10¹³ + 223,999.9492…] |
| **Ordinal range** | #43,124,192,297,104 through #43,124,193,298,735 |
| **Zeros located** | 1,001,632 |
| **Zeros required (via `nzeros`)** | 1,001,632 |
| **Shortfall after refine** | 0 (100.0000% complete) |
| **Suspected RH violations** | 0 (no candidate survived mpmath verification past dps=50) |
| **Tight-pair near-misses recorded** | 145 (all with `norm_gap` < 0.05) |
| **Tightest pair (normalized gap)** | norm_gap = 0.01124, gap = 2.513×10⁻³, extremum Z ≈ −7.0×10⁻⁵ |
| **Tightest pair location** | t ≈ 10,000,000,097,148.81 |
| **Tightest pair ordinals** | #43,124,192,731,510 and #43,124,192,731,511 |
| **Sweep chunks** | 224 (chunks 0 through 223, each covering 1000 units of t) |

The tightest-pair result is consistent with GUE random-matrix predictions: for a sample of ~10⁶ zeros, the expected minimum normalized spacing scales as roughly N⁻¹/³, giving ~0.010–0.015, and the observed value 0.01124 falls squarely in that range.

---

## Method

The scanner combines classical Riemann–Siegel with GPU parallelism, high-precision argument reduction, and independent completeness certification. Seven-step summary:

1. **Chunk-based sweep.** The t range is divided into chunks of `CHUNK_T = 1000` units. Each chunk samples Z(t) on a grid of step `STEP = 0.01`, yielding ~100,000 samples per chunk.

2. **GPU-parallel main sum.** The Riemann–Siegel main sum, Z(t) = 2 · Σₙ cos(θ(t) − t·ln n) / √n for n = 1..N with N = ⌊√(t/2π)⌋, is computed for the whole chunk's grid in a single CUDA operation (float64). At t = 10¹³, N ≈ 1,261,566 terms per sample.

3. **High-precision argument reduction.** At height 10¹³ the cosine argument t·ln n reaches ~10¹⁴, and native float64 quantizes past the sample step (ULP ≈ 2×10⁻³). The code uses a `(t₀, dt)` grid split: the anchor `t₀` is held at full mpmath precision and only the small offset `dt` participates in float64 arithmetic. This preserves argument-reduction precision at any height the algorithm can otherwise reach. See the `exact_t` helper and the residue-worker Stage 3 machinery in `zeta_gpu_scan_v6_hp.py`.

4. **Riemann–Siegel remainder from mpmath.** The R(t) correction is computed by `mpmath.siegelz` at anchor points across the chunk (interpolated to grid points), not derived from the GPU float64 result. This is an independent ground truth for the correction, so a bug in the GPU main sum could not silently corrupt the reported Z(t) values.

5. **Completeness check via `nzeros`.** After the coarse sweep counts sign changes across the chunk, the code invokes `mpmath.nzeros(t_b) − mpmath.nzeros(t_a)` and compares. If the located count is short, the chunk is re-swept at 10× denser resolution (`FINE_STEP = 0.001`) to catch tight pairs the coarse grid missed. Every chunk marked `[OK]` in the log has passed this comparison. To further reduce cost, the two boundary `nzeros` calls run asynchronously on a dedicated worker pool alongside the GPU compute, so the wall-time impact is minimal.

6. **Extremum extraction and violation verification.** Turning points of Z(t) with |Z| < `LEHMER_THR = 0.02` are candidates for near-miss pairs. A parabolic fit through three grid samples estimates the extremum. Any "wrong-side" extremum (a positive local minimum, or a negative local maximum — which would indicate Z crossing zero somewhere near) is verified by `mpmath.siegelz` at dps=50. If it survives that, it escalates to dps=100 and then dps=200 before being reported as a candidate RH violation. In this dataset, zero candidates survived past dps=50.

7. **Precise gap measurement.** For each tight near-miss (small normalized gap), the two zeros bracketing the extremum are located by parallel outward scan and parallel mpmath bisection, producing gap measurements accurate to ~10⁻¹¹.

Every zero also gets its **true ordinal index** (its position in the Riemann zero sequence) computed from `nzeros(t_start) + local_position`. This is what makes the data joinable to any other zero catalog by integer key.

---

## Provenance

| Item | Value |
| :--- | :--- |
| **Author** | Franklin Marek ([SetiAstro](https://setiastro.com)) |
| **Code version** | [`zeta_1e13+224kT`](https://github.com/setiastro/zeta_sweep/releases/tag/zeta_1e13%2B224kT) release tag |
| **Hardware** | Xeon E5-2696 v3 (18 cores / 36 threads), 256 GB RAM, NVIDIA RTX 3070 Ti (8 GB) |
| **GPU compute** | float64 CUDA (available on all consumer NVIDIA cards; datacenter cards run substantially faster) |
| **Software** | Python 3.10+, PyTorch (CUDA), mpmath, NumPy |
| **Verification precision** | mpmath dps=50 for standard checks; escalated to dps=100 and dps=200 for suspected off-critical-line extrema |
| **Completeness method** | Two-endpoint `nzeros` check plus per-chunk `nzeros` checks during the sweep (see Completeness section) |
| **Independence** | Not derived from any other zero catalog. Recomputed from ζ(s) via Riemann–Siegel. |
| **Location** | Madisonville, Louisiana, USA |

---

## Completeness

Every zero in the interval t ∈ [10¹³, 10¹³ + 223,999.95] is present in the catalog.

**How verified.** For each of the 224 chunks in the sweep, the code independently computes `nzeros(t_a)` and `nzeros(t_b)` at the chunk's endpoints (using `mpmath`'s implementation of the Riemann–von Mangoldt / Turing zero-counting formula). The number of sign changes of Z(t) located in the chunk's interior must equal `nzeros(t_b) − nzeros(t_a)`. If it doesn't, the chunk is re-swept at 10× denser resolution. All 224 chunks passed this check with zero shortfall after any needed refinement.

**Additional end-to-end verification.** In addition to the per-chunk checks, a two-endpoint verification was performed: `nzeros(t_start_of_run)` and `nzeros(t_end_of_run)` were computed independently, and their difference exactly equals the total row count of the zeros CSV (1,001,632). This provides a single-integer integrity check on the entire catalog.

**How to independently verify.** Anyone with `mpmath` installed can spot-check this claim in seconds:

```python
from mpmath import mp, mpf, nzeros
mp.dps = 50
# Zeros in the swept interval, as counted independently:
n_end   = int(nzeros(mpf("10000000223999.95")))
n_start = int(nzeros(mpf("10000000000000.252")))
print(f"nzeros difference: {n_end - n_start:,}")   # → 1,001,632
```

If that number matches, the completeness claim holds. Every row in `zeta_zeros_v6_1p0e13.csv` corresponds to exactly one zero in the interval, in true t-order.

**Known limitations.**
- **Distributed contributions have small overlap.** If workers are assigned adjacent chunk ranges, the "safe boundary" mechanism means each worker's boundary is nudged inward until |Z| > 0.20, so adjacent workers' boundaries may cover slightly overlapping regions. Zeros in the overlap may be present in both workers' outputs. A merge script to dedupe by ordinal (planned) would resolve this cleanly.
- **The single-run catalog reported here does not have this issue** — one continuous run has no seam overlap.

---

## Reliability

**The precision guarantee.** The reported zero locations are known to lie on the critical line to the precision of the calculation — specifically, to the precision at which `mpmath.siegelz` was evaluated during verification (dps=50 for routine checks; dps=100 or dps=200 for any extremum that appeared wrong-sided at lower precision).

**What this is NOT.** This is not the same as an interval-arithmetic-rigorous verification. Platt & Trudgian's 2020 work computed certified enclosures on Z(t) using ARB / interval arithmetic, so every claim about a zero's location comes with a machine-verified error bound. This project uses arbitrary-precision floating-point arithmetic in mpmath, which is highly reliable in practice but does not produce formally-certified enclosures. In particular:

- A bug in mpmath's `siegelz` or `nzeros` implementation would silently propagate into this dataset. (These functions have been widely used for two decades and no such bug has been discovered, but the possibility cannot be ruled out formally.)
- Roundoff errors in the GPU main sum are bounded by float64's ~10⁻¹⁶ relative precision, but at the sample-count scale we use (~10⁶ terms summed per chunk sample), accumulated error could reach ~10⁻¹⁰ in principle. Cross-checks vs. mpmath show stability at the ~10⁻⁶ level empirically.
- The 145 tight-pair near-misses all had gaps well above the numerical noise floor, so the tight-pair record is robust to any plausible numerical error.

**For strict rigor at heights up to ~3 × 10¹²**, use Platt & Trudgian's results, which are the current published gold standard. For a computational catalog past that frontier — with high but not certified confidence — this project fills a niche that hasn't been filled before at these specific heights.

**How to spot-check a specific zero.** Any row in the zeros CSV can be independently verified:

```python
from mpmath import mp, mpf, siegelz, nzeros
mp.dps = 50

# Pick a zero, e.g., the tightest-pair extremum's left neighbor:
t = mpf("10000000097148.813")   # actual t from the CSV
z = float(siegelz(t))
print(f"Z(t) = {z:+.4e}")   # should be ~0 (magnitude ~1e-4 or smaller)

# Confirm ordinal index:
n = int(nzeros(t))
print(f"nzeros(t) = {n:,}")   # should match the CSV's zero_index for that row
```

**Validation ladder.** The scanner has been re-validated at each height during development:

| Test point | Result |
| :--- | :--- |
| t ≈ 7005.082 | Reproduces the classic Lehmer pair: `gap = 3.769850e-02`, `norm_gap = 0.04210`, extremum Z = +0.003967 (max) |
| t = 2.5 × 10¹⁰ | 100+ chunks, `nzeros` match exact on every chunk |
| t = 10¹² | 100 chunks, 410,511 / 410,511 zeros, no shortfall |
| t = 10¹³ | 224 chunks, 1,001,632 / 1,001,632 zeros, no shortfall, no violations — **this dataset** |

---

## Data schemas

### `zeta_zeros_v6_1p0e13.csv` (bulk catalog, in release)

```
t, Z_left, Z_right, dt_left, dt_right, chunk, zero_index
```

| Column | Description |
| :--- | :--- |
| `t` | Interpolated absolute zero location. Float64 precision (ULP ≈ 2×10⁻³ at 10¹³, which is the storage-format limit). |
| `Z_left`, `Z_right` | Z values at the two straddling grid samples. Small magnitudes indicate either precise interpolation or a tight-pair neighborhood. |
| `dt_left`, `dt_right` | Grid-sample offsets bracketing this zero. `dt_right − dt_left ≈ 0.01` for a normal coarse-grid zero; `≈ 0.001` for a zero recovered via the fine resweep (higher-precision provenance, effectively free). |
| `chunk` | Chunk index within the run. |
| `zero_index` | **Ordinal position in the Riemann zeta zero sequence** (`nzeros(t_at_first_zero_of_first_chunk) + local_index`). Exact integer. |

### `zeta_hits_v6_1p0e13.csv` (near-misses, in repo)

```
t, Z_mpmath, kind, gap, norm_gap, zero_index
```

| Column | Description |
| :--- | :--- |
| `t` | Location of the near-miss extremum (parabolic vertex, float64). |
| `Z_mpmath` | Extremum value evaluated at high precision (dps=50). Small \|Z\| indicates axis-hugging extremum. |
| `kind` | `max` or `min`. |
| `gap` | Precise measurement of the spacing between the two zeros bracketing this extremum, from parallel mpmath bisection (~10⁻¹¹ precision). Blank for hits that didn't trigger precise-gap refinement. |
| `norm_gap` | `gap` divided by local mean spacing (2π / ln(t/2π)). Small values are the interesting ones (Lehmer-like near-misses). |
| `zero_index` | Ordinal of the LEFT zero of the pair. Right zero is `zero_index + 1`. |

### `zeta_scan_v6_1p0e13.json` (checkpoint, in repo)

Cumulative counts, resume state, and both leaderboards (top-50 tightest by |Z| at extremum; top-50 tightest by normalized gap). Both leaderboards carry `zero_index` on every entry.

---

## Files in the release vs. the repo

**In the release [`zeta_1e13+224kT`](https://github.com/setiastro/zeta_sweep/releases/tag/zeta_1e13%2B224kT):**

| File | Description |
| :--- | :--- |
| `zeta_zeros_v6_1p0e13.csv` | The bulk catalog — every zero located, one row each, with ordinal. Too large for git-tracked storage; distributed as a release asset. |

**In the repo (`main` branch):**

| File | Description |
| :--- | :--- |
| `zeta_gpu_scan_v6_hp.py` | The scanner engine. Runs headless from CLI; also drives the UI via subprocess. |
| `zeta_scanner_ui.py` | PyQt6 desktop UI wrapper. Start / Pause / Resume / Abort with live status and output pane. |
| `zeta_hits_v6_1p0e13.csv` | Near-miss leaderboard from this sweep (small enough to ship in-repo). |
| `zeta_scan_v6_1p0e13.json` | Checkpoint from this sweep, with resume state and top-50 leaderboards. |
| `requirements.txt` | Python dependencies (numpy, mpmath, torch, PyQt6). |
| `LICENSE` | MIT. |
| `README.md` | This file. |

Later releases will bundle expanded sweeps (e.g., additional chunks past chunk 223, or fresh runs at higher `T_BASE`) as they complete.

---

## Requirements and installation

- Python 3.10 or later
- CUDA-capable NVIDIA GPU with float64 support (RTX 30-series or newer recommended; datacenter cards like A100/H100 will be substantially faster)
- Python packages: install with `pip install -r requirements.txt`

**Important note on PyTorch:** the default `pip install torch` may install a CPU-only build or the wrong CUDA version. Instead, visit [pytorch.org](https://pytorch.org) and use the selector to get the correct wheel URL for your CUDA version (e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu121` for CUDA 12.1). The `requirements.txt` file has details.

**Performance reference** (on the hardware above): sustained rate of ~4 units of t per second on normal chunks at t = 10¹³, corresponding to ~17,900 zeros per hour. One million zeros in continuous operation is roughly 60 hours of wall time (~2.5 days). A rig with datacenter GPUs would run this substantially faster.

---

## Running headless

```bash
python zeta_gpu_scan_v6_hp.py
```

Runs the module-default config (edit `T_BASE`, `N_CHUNKS`, and filenames at the top of the file, or override via CLI flags). Outputs a checkpoint JSON, a hits CSV, a zeros CSV, and a count log to the current directory.

**Optional CLI flags** (all composable, all optional):

```
--t-base VAL            Override module T_BASE (e.g. 1e13)
--chunk-t VAL           Override CHUNK_T (dt units per chunk)
--n-chunks N            Upper bound on chunk index (run exits after this)
--start-chunk N         Force starting chunk (ignore resume checkpoint)
--output-prefix PREFIX  Rename output files: {PREFIX}_zeros.csv, etc.
--json-progress         Emit structured @@STATUS@@ JSON events (for UI/coordinator use)
--pause-flag PATH       Between chunks, exit cleanly if this file exists
```

**Example — split a run for a distributed contribution:**

```bash
# Worker A: chunks 200..299 at 1e13
python zeta_gpu_scan_v6_hp.py --t-base 1e13 --start-chunk 200 --n-chunks 300 --output-prefix chunks_200_300

# Worker B: chunks 300..399 at 1e13
python zeta_gpu_scan_v6_hp.py --t-base 1e13 --start-chunk 300 --n-chunks 400 --output-prefix chunks_300_400
```

Each worker's four files (`{prefix}_zeros.csv`, `_hits.csv`, `_count.csv`, `_checkpoint.json`) can be merged post-hoc. A merge script is planned.

## Running with the UI

```bash
python zeta_scanner_ui.py
```

A single window with:
- Configuration fields (scanner script path, T_BASE, N_CHUNKS, start-chunk, output prefix, pause flag path, working directory) with per-field override checkboxes — all persisted across launches via QSettings
- State-aware **Start / Pause / Resume / Abort** buttons
- Progress bar with chunk N of M, elapsed, ETA
- Cumulative statistics: zeros located, required, shortfall after refine, violations
- **Tightest pair widget** showing left/extremum/right t values with matching-digit alignment (Decimal arithmetic so float64 quantization at 10¹³ doesn't lie), normalized gap, precise gap, extremum Z, kind, and ordinal indices of both zeros in the pair
- Three-way rate display: instantaneous, 10-chunk rolling average, and cumulative-since-launch
- Scrollable live-output pane with sub-phase timing
- Reset-to-defaults button (with confirmation)

Pause is cooperative — the scanner finishes the current chunk, saves its checkpoint, and exits cleanly. Resume relaunches and picks up exactly where it stopped. Abort terminates the subprocess; previously completed chunks remain safely checkpointed.

---

## Loading the data

CSVs work with anything that isn't Excel (Excel caps at ~1M rows and struggles with 80 MB files; use pandas / numpy / SQLite / awk instead):

```python
import pandas as pd

# Bulk catalog (downloaded from the release)
zeros = pd.read_csv("zeta_zeros_v6_1p0e13.csv")
print(f"{len(zeros):,} zeros in t ∈ [{zeros.t.min():.0f}, {zeros.t.max():.0f}]")
print(f"Ordinals #{zeros.zero_index.min():,} .. #{zeros.zero_index.max():,}")

# Near-misses (in repo)
hits = pd.read_csv("zeta_hits_v6_1p0e13.csv")
tight = hits.dropna(subset=["norm_gap"]).nsmallest(10, "norm_gap")
print("Ten tightest normalized gaps:")
print(tight[["t", "norm_gap", "gap", "zero_index"]])
```

---

## Reproducing the tightest-pair record

The tightest pair in this dataset is between zeros #43,124,192,731,510 and #43,124,192,731,511, at t ≈ 10,000,000,097,148.81. To confirm this independently:

```python
from mpmath import mp, mpf, siegelz, nzeros
mp.dps = 50

# The extremum sits between two zeros
t_ext = mpf("10000000097148.814")
z_ext = float(siegelz(t_ext))
print(f"Z at extremum: {z_ext:+.4e}")   # small negative value ~-7e-5

# Confirm the two zeros bracket it
t_a = mpf("10000000097148.80")
t_b = mpf("10000000097148.83")
n_diff = int(nzeros(t_b) - nzeros(t_a))
print(f"Zeros in [{t_a}, {t_b}]: {n_diff}")   # → 2

# The ordinal of the left zero
n_at_extremum = int(nzeros(t_ext))
print(f"Left zero ordinal: #{n_at_extremum:,}")   # → 43,124,192,731,510
```

---

## Contributing chunks

If you want to help extend this catalog further: open an issue on the repo to coordinate a chunk range at some agreed T_BASE, then run the scanner as above with `--start-chunk`, `--n-chunks`, and `--output-prefix` set for your assigned range. Send back your four output files and they can be merged into the main catalog.

**Good contribution ranges** to consider:
- **Extend the current sweep upward** — chunks 224 onward at T_BASE = 10¹³. Each additional chunk adds ~4,472 zeros to the catalog.
- **Sample at higher heights** — a fresh run at T_BASE = 10¹⁴ would let us compare tight-pair statistics across heights (a test of the height-invariance predicted by random-matrix theory).
- **Rerun a covered range with independent code** — an independent verification of the current dataset using a different tool (Arb, ANTIC, etc.) would be genuinely valuable, especially for the tightest-pair record.

---

## Computation Contributors

-David Chaton
-Martin Gibbs - Other published works by Gibbs: DOI: 10.1016/j.bios.2014.09.036 https://www.sciencedirect.com/science/article/abs/pii/S0956566314007234

## Explicit non-claims

- **Zeros outside the swept range are not addressed.** This dataset says nothing about any zero of ζ(s) with imaginary part outside [10¹³, 10¹³ + 223,999.95].
- **This is not a proof or disproof of RH.** The Riemann Hypothesis is an infinite statement; verifying it for any bounded interval is finite evidence, not proof. For all zeros in the swept range, no counterexample was found.
- **The precision guarantee is via independent mpmath cross-verification and `nzeros` completeness, not interval arithmetic.** For a stricter interval-arithmetic result up to ~3 × 10¹², use Platt & Trudgian (2020).
- **Precise-gap measurements are to ~10⁻¹¹.** Bulk-catalog zero locations are to float64 storage precision (~10⁻³ absolute at t = 10¹³, which is the storage-format limit rather than the algorithm's).
- **Ordinal indices are exact.** They come from `mpmath.nzeros`, which is an integer-valued function; the two-endpoint check confirmed `n_end - n_start` matches the CSV row count exactly.

---

## License

MIT — see `LICENSE`. Data in the releases is also under MIT. Attribution is appreciated but not required.

## Citation

If you use this data or code, a citation like the following is appreciated:

> Marek, F. (SetiAstro). *zeta_sweep: GPU-accelerated Riemann–Siegel Z-function sweep with ordinal-indexed completeness certification.* GitHub, 2026. https://github.com/setiastro/zeta_sweep. Dataset: release `zeta_1e13+224kT`.

The release tag identifies the exact range and code version, so specific claims can be traced to specific dataset versions.

## Acknowledgments

Built on top of `mpmath`'s excellent implementations of `siegelz`, `siegeltheta`, and `nzeros`. Method inspired by the classical Riemann–Siegel approach combined with modern GPU parallelism for the main sum. Independent verification of the 2020 Platt–Trudgian frontier at heights past 10¹² was the motivating goal; the ordinal-indexed structure was added later so the data would be joinable to other zero catalogs.
