# AirDbM-API-Bench

> <kbd> Aug 5, 2026 </kbd> <br> This is the native-Python **API** and **benchmark** release of the AirDbM design and evaluation scheme. The parent repository is [AirDbM](https://github.com/UCBCFD/AirDbM) — the MATLAB implementation accompanying the article that introduced the compact 12-baseline Design-by-Morphing set this package builds on.

<div align="center">
  <kbd>
    <img width="640" alt="barycentric_airdbm" src="https://github.com/user-attachments/assets/bb92cbfc-e583-450d-9570-537c9ae0de72"/>
  </kbd> 
</div><br/>

~~~python
import numpy as np
from airdbm_core import TestAirfoils

x = np.array([[2/3] * 3]) # Equal-weight interpolative morphing
results = TestAirfoils(x) # Call the AirDbM API (design & eval)

cl_cd_max, _ = results[0].objectives
print(f"Cl/Cd max: {cl_cd_max:.2f}") # >>>>>>> Cl/Cd max: 54.98
~~~

This repository provides a highly robust, parallelized Python interface for generating morphed airfoil geometries using Design-by-Morphing (DbM) and evaluating them dynamically via XFOIL. Two modules, each named for what it carries:

| module | carries |
|---|---|
| `airdbm_core.py` | the engine: DbM geometry generation, Shapely geometry repair, containerized XFOIL evaluation, parallel batch evaluation |
| `airdbm_service.py` | the JSON/HTTP service: FastAPI app, request/response schemas, benchmark argument presets, the `airdbm-serve` entry point |

The benchmark ships alongside it in `bench/`:

| path | carries |
|---|---|
| `bench/problems.py` | the twelve problem specs and the frozen evaluation arguments |
| `bench/run.py` | one (problem, method, seed) run; writes its evaluation history |
| `bench/gate.json` | the frozen reference set that arms the in-loop stability gate |
| `bench/data/` | released evaluation histories, the two summaries, the problem manifest |
| `bench/DATA.md` | layout and schema of everything under `bench/data/` |



This API is reprocessed from the original [AirDbM](https://github.com/UCBCFD/DbMAirfoilOpt) repository and adapted into a standalone Python workflow for robust batch evaluation.

If this repository contributes to your research, publication, or benchmark, we kindly ask that you credit our work by citing our AirDbM research references:
- Lee, S. & Sheikh, H. M. (2026). Airfoil Optimization using Design-by-Morphing with Minimized Design-Space Dimensionality. *Journal of Computational Design and Engineering*, 13(1), 108-124. [![DOI](https://img.shields.io/badge/DOI-10.1093%2Fjcde%2Fqwaf124-blue)](https://doi.org/10.1093/jcde/qwaf124)
- Sheikh, H. M., Lee, S., Wang, J., & Marcus, P. S. (2023). Airfoil Optimization using Design-by-Morphing. *Journal of Computational Design and Engineering*, 10(4), 1443–1459. [![DOI](https://img.shields.io/badge/DOI-10.1093%2Fjcde%2Fqwad059-blue)](https://doi.org/10.1093/jcde/qwad059)

## Benchmark Database

Beyond the API, this repository doubles as **AirDbM-Bench**, a real-world
(physics-in-the-loop) optimization benchmark database: **12 frozen airfoil optimization
problems** (single- and bi-objective; input dimension `D ∈ {4, 8, 12}`; two physically
realizable flight conditions) with **released reference solutions and complete per-evaluation
optimization histories**. Unlike synthetic suites (ZDT/DTLZ) or closed-form real-world suites (RE/MODAct),
every objective evaluation is a live XFOIL solve, so an optimizer is tested in a genuinely
expensive, simulation-bound regime. Every problem's evaluation budget is scaled with its dimension as
`1024 · D`, i.e. 4096/8192/12288 evaluations at `D` = 4/8/12, over 5 seeds per optimizer — so a
comparison at fixed budget is a comparison at fixed budget-per-dimension.

The two conditions pair Mach with Reynolds number as they actually occur, so each is a real
operating point rather than a coordinate in a sweep:

| condition | Ma | Re_c | physical realization |
|---|---|---|---|
| wind-tunnel scale model | 0.20 | 1e6 | 215 mm chord at 68.1 m/s, sea level |
| regional turboprop cruise | 0.40 | 1e7 | 1.93 m chord at 126.4 m/s at 20 000 ft (Saab 340B / ATR 42 class) |

Both are subcritical and shock-free, i.e. inside the envelope where XFOIL's formulation is valid.
Because the two differ in *both* Ma and Re_c, a difficulty difference between them reflects the
change of operating point as a whole and should not be attributed to either variable alone.

Problems are named `ADO-<O>-<C>-<N>`: `O` = objective form (`S` single / `M` multi), `C` = flow
condition keyed by Mach (`2` = Ma 0.2, `4` = Ma 0.4), `N` = dimension index (`1` = D 4, `2` = D 8,
`3` = D 12).

Reading the released data needs no XFOIL and nothing from this repository — the histories are plain
gzipped CSV, one file per (optimizer, seed), one row per evaluation in evaluation order:

~~~python
import json, numpy as np

# every evaluation NSGA-II made on ADO-M-2-1, in order
a = np.loadtxt("bench/data/MO_D4_Re1e+06_Ma0.2/nsga2_seed0.csv.gz",
               delimiter=",", skiprows=1)
X, Y, curve = a[:, :4], a[:, 4:6], a[:, -1]   # designs, objectives, running hypervolume

# the released reference front and per-optimizer attainment
mo = json.load(open("bench/data/mo_summary.json"))["per_problem"]
front = np.array(mo["ADO-M-2-1"]["front_Y"])
mo["ADO-M-2-1"]["hv_ref_norm"]                # normalize your own hypervolume against this
~~~

### Running your own optimizer against a problem

A problem is a frozen bundle of evaluation arguments plus a budget, so any optimizer that can call a
batch function can be scored on it. Complete working example with pymoo's NSGA-II — the only
benchmark-specific lines are `spec` (the frozen condition), `evaluate` (the XFOIL batch) and `screen`
(the stability gate every released evaluation passed):

~~~python
import sys, json
import numpy as np

sys.path.insert(0, "bench")
from problems import spec, evaluate, screen

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

s = spec("ADO-M-2-1")            # bi-objective, D = 4, Ma 0.20 / Re_c 1e6, budget 4096
WORKERS = 32                     # XFOIL solves run in parallel

class ADO(Problem):
    def __init__(self):
        super().__init__(n_var=s["D"], n_obj=s["m"], xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        Y = evaluate(X, s, max_workers=WORKERS)              # one real XFOIL solve per row
        Y, _ = screen(X, Y, s, max_workers=WORKERS)          # the suite's stability gate
        out["F"] = -np.asarray(Y, float).reshape(-1, s["m"])  # pymoo minimizes; both objectives are maximized

pop = 100
res = minimize(ADO(), NSGA2(pop_size=pop), ("n_gen", s["budget"] // pop), seed=0, verbose=True)
Y = -res.F                       # your front: [(Cl/Cd)_max, stall margin]

# score it the way the released summaries do
ref = json.load(open("bench/data/mo_summary.json"))["per_problem"]["ADO-M-2-1"]
z = np.array(ref["front_Y"]).max(axis=0)
score = float(HV(ref_point=np.array([0.05, 0.05]))(-(Y / z))) / ref["hv_ref_norm"]
print(f"fraction of the reference hypervolume: {score:.3f}")
~~~

`score` is directly comparable to the `frac_of_reference_hv` values in `mo_summary.json`, where NSGA-II
reaches about `0.93` on this problem at the same budget. For the single-objective problems use `n_obj=1` and compare the
best `y_cl_cd` against `y_ref` in `so_summary.json`. XFOIL must run on a compute node, not a login node,
and a full-budget run is hours of wall time: start with a small `n_gen` to check the wiring.

See **[`bench/DATA.md`](bench/DATA.md)** for the directory layout, the column schema of both objective
forms, the summary-file fields, and how to compare your own optimizer against the references.

## Setup & Installation

1. Preliminary requirements:
- Python (tested with `3.10.12`)
- Apptainer (https://apptainer.org/docs/)

2. Install the required Python packages:
~~~bash
pip install -r requirements.txt
~~~

3. Build the Apptainer image for isolated XFOIL execution:
~~~bash
make xfoil-apptainer-build
~~~
*(Verify the container functions correctly by running `make xfoil-apptainer-check`)*

By default, `TestAirfoils` uses the Apptainer backend (`xfoil_backend='apptainer'`) and the Ubuntu 22.04-based XFOIL image built from the container definition (`bin/containers/xfoil-ubuntu22.def`) in this repository. This is the recommended mode for reproducibility and consistency across systems.

You may choose native XFOIL execution by setting `xfoil_backend='native'`.

## How to Use the API

Import `TestAirfoils` from `airdbm_core.py` into your optimization loop or script.

### Function Signature

~~~python
TestAirfoils(x: np.ndarray, args: dict | None = None, m: int = 2) -> list
~~~

### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `x` | `np.ndarray` | Required | Candidate matrix of shape `N x D` (`N` candidates with `D` parameters); each entry is in `[0.0, 1.0]`. |
| `args` | `dict` | `{}` | Configuration dictionary for DbM generation, parallelism, and XFOIL execution (see table below). |
| `m` | `int` | `2` | Number of objectives to return when XFOIL is enabled. Supported: `1` or `2`. |

### `args` Configuration Reference

**AIRFOIL DESIGN ARGS**

| Key | Type | Default | Description |
|---|---|---|---|
| `airfoil_db_dir` | `str` | `'airfoilDB'` | Path to the airfoil database folder. |
| `dbm_baselines` | `list[str]` | Internal `EXPECTED_BASELINES` list | Baseline airfoil names. First `D` entries are used for a candidate with `D` parameters. The internal list is based on the optimal baseline set reported in our [2026 paper](https://doi.org/10.1093/jcde/qwaf124). |
| `dbm_weight_range` | `list[float]` | `[-1.0, 1.0]` | Maps input `x` linearly from `[0.0, 1.0]` to morphing weights. |
| `dbm_normalization` | `str \| None` | `None` | Weight normalization mode: `None`, `'SUM'`, or `'ABS_SUM'`. |

**AIRFOIL EVALUATION ARGS**

| Key | Type | Default | Description |
|---|---|---|---|
| `xfoil_evaluation` | `bool` | `True` | If `False`, returns generated `Airfoil` objects without aerodynamic evaluation. |
| `xfoil_backend` | `str` | `'apptainer'` | XFOIL execution backend: `'apptainer'`, `'native'`, or `'auto'`. |
| `xfoil_iter` | `int` | `200` | Max XFOIL iterations per alpha step. |
| `xfoil_timeout` | `float` | `60.0` | Timeout (seconds) per run by default. Set `0.0` for no timeout. |
| `xfoil_retry` | `int` | `1` | Number of XFOIL reattempts for a candidate. This helps recover transient deadlock/no-parse behavior in multiprocessing, where a design may normally converge in another attempt. |
| `xfoil_strict` | `bool` | `True` | If `True`, raise on XFOIL errors; otherwise attach errors in the result payload. |
| `alfa_start` | `float` | `0.0` | Start angle of attack (deg) for scans. |
| `alfa_end` | `float` | `45.0` | End angle of attack (deg) for scans. |
| `reynolds` | `float` | `1e6` | Reynolds number for viscous analysis. |
| `mach` | `float` | `0.0` | Mach number; `0.0` indicates a negligible-compressibility assumption. |
| `n_crit` | `float` | `9.0` | e^N transition amplification factor. |
| `clcd_ceiling` | `float` | `350.0` | Hard upper clip on `Cl/Cd`, applied at every Reynolds number, guarding against solver blow-ups. The benchmark objective is `min(Cl/Cd_max, 350)`. |

**MULTIPROCESSING ARGS**

| Key | Type | Default | Description |
|---|---|---|---|
| `parallel` | `bool` | `True` | Enables multiprocessing across candidates. |
| `max_workers` | `int` | `os.cpu_count()` | Maximum worker processes (capped by available CPUs). |

### Return Value

`TestAirfoils` always returns a list of `AirfoilEvaluationResult` objects (length `N`), where each element contains:

| Field | Type | Description |
|---|---|---|
| `.airfoil` | `Airfoil` | Generated morphed airfoil object (always available). |
| `.xfoil_result` | `dict \| None` | Raw XFOIL metrics dictionary when `xfoil_evaluation=True`; otherwise `None`. |
| `.objectives` | `None \| float \| list[float]` | `None` if `xfoil_evaluation=False`; with evaluation enabled: `m=1 -> Cl/Cd_max`, `m=2 -> [Cl/Cd_max, delta_alpha]`. |

The `.airfoil` object exposes both geometry data and helper methods, for example:

- metadata: `.airfoil.name` (DbM weight inputs by default)
- geometry arrays: `.airfoil.x_raw`, `.airfoil.y_raw`
- access helpers: `.airfoil.get_raw_coordinates()`
- quick visualization: `.airfoil.plot(save_path=...)`

As for the objectives,

$$
\left(\frac{C_l}{C_d}\right)_{\max}
= \max_{\alpha}\left(\frac{C_l(\alpha)}{C_d(\alpha)}\right)
$$

This is the maximum lift-to-drag ratio over the evaluated angle-of-attack sweep.

$$
\Delta\alpha = \alpha_{\mathrm{stall}} - \alpha_{\left(\frac{C_l}{C_d}\right)_{\max}}
$$

Here, $\alpha_{\mathrm{stall}}$ is identified as the first local maximum of $C_l$ encountered while marching upward in angle of attack, starting from $\alpha_{\left(C_l/C_d\right)_{\max}}$ (the peak lift-to-drag angle). Because the search begins at the peak-efficiency angle, the stall margin $\Delta\alpha$ is non-negative by construction.

### Verifying a design before you publish it as a reference

The objective is a deterministic function of the design vector, but determinism is not stability:
XFOIL can hold two different boundary-layer solutions for sections that are geometrically
indistinguishable (a long laminar run versus a transitioned one), so an isolated design vector can
score far above every design around it. Such a value is reproducible yet unreachable by search, and
it should not define a reference optimum or a Pareto front.

`VerifyDesigns` screens for exactly that. It perturbs each design in `n_dir` random directions at
radius `eps` and reports whether the neighbors agree with it:

~~~python
import numpy as np
from airdbm_core import VerifyDesigns

x_best = np.array([[2/3, 2/3, 2/3]])           # the designs you are about to publish

v = VerifyDesigns(x_best, m=2)                 # -> one verdict dict per design
[(r['robust'], round(max(r['rel_dev']), 4)) for r in v]
# >>>>>>> [(True, 0.0)]
~~~

| Argument | Default | Meaning |
|---|---|---|
| `eps` | `1e-6` | perturbation radius in design space |
| `directions` | `"axes"` | `"axes"` probes `x ± eps·e_i` on every coordinate, so the verdict is a function of the design vector alone; `"random"` draws `n_dir` directions per design instead |
| `n_dir` | `4` | perturbed neighbors per design, used only when `directions="random"` |
| `rel_tol` | `0.02` | relative deviation tolerated on the deciding objective |
| `objective` | `0` | which objective decides the verdict |
| `seed` | `0` | fixes the random directions, so a verdict is reproducible |

Each verdict carries `robust`, the per-objective `rel_dev` and `abs_dev`, the `median_neighbor` it was
compared against, `rel_dev_quantiles` and `frac_within_tol` for re-thresholding without re-evaluating,
and `n_informative` — how many neighbors converged. A design is `robust` when at least two neighbors
converged and the median one agrees with it to within `rel_tol` on the deciding objective.

This is deliberately **not** part of `TestAirfoils`: it costs `1 + n_dir` evaluations per design, so
running it inside the objective would multiply the cost of an optimization run. Apply it to the
O(front size) candidate reference set at the end of a study instead, which is how the released
reference solutions were screened. Pass the same `args` the design was optimized under: a benchmark
condition is a whole bundle (including `dbm_normalization` and `dbm_weight_range`), not just Mach and
Reynolds, so a partial `args` verifies a different geometry than the one you optimized.

### Python Script Example

~~~python
import numpy as np
from airdbm_core import TestAirfoils

rng = np.random.default_rng(seed=32)

# Example: 4 design candidates (N=4), using 12 input parameters (D=12).
candidate_weights = rng.uniform(0.0, 1.0, size=(4, 12))

# Override some default arguments based on needs
args = {
    'dbm_weight_range': [-1.0, 1.0], # allow both interpolative and extrapolative morphing during DbM
    'dbm_normalization': 'ABS_SUM', # enforce L1-style normalization: sum(|w_i|) = 1 across baselines
    'reynolds': 1e5, # override the Reynolds number (default is 1e6)
}

# Run the API for Bi-Objective evaluation (m=2)
results = TestAirfoils(candidate_weights, args=args, m=2)

for i, res in enumerate(results):
    cl_cd_max, delta_alpha = res.objectives
    print(
        f"Candidate {i+1} ({res.airfoil.name}) -> "
        f"Cl/Cd max: {cl_cd_max:.2f}, Stall Margin (deg): {delta_alpha:.2f}"
    )
~~~

## HTTP API server (JSON)

Besides the in-process `TestAirfoils` call, the same evaluator is exposed as a JSON/HTTP service
(FastAPI + Pydantic, with an auto-generated OpenAPI 3.1 schema), so an optimizer written in any
language can drive it. `airdbm_core.py` is unchanged — the service only imports from it.

~~~bash
pip install -e ".[server]"        # installs fastapi/uvicorn/pydantic + the airdbm-serve script
airdbm-serve --host 0.0.0.0 --port 8000
# interactive docs at /docs ; machine-readable schema at /openapi.json
~~~

| method | path | purpose |
|---|---|---|
| `GET`  | `/healthz` | liveness probe |
| `GET`  | `/v1/meta` | versions, defaults, objective modes, limits |
| `GET`  | `/v1/baselines` | the ordered DbM baseline library |
| `GET`  | `/v1/presets` | **named argument presets** for the benchmark conditions |
| `POST` | `/v1/evaluate` | morph + evaluate a batch of design vectors |
| `GET`  | `/v1/benchmark/problems` | the 12 ADO benchmark problems |
| `GET`  | `/v1/benchmark/problems/{id}` | one problem: parameters + released reference solution |
| `POST` | `/v1/benchmark/problems/{id}/evaluate` | evaluate at a problem's frozen condition |

**Argument presets.** Rather than restating Mach, Reynolds and the Cl/Cd guard on every call — and
risking a typo that silently scores a design under the wrong condition — name the benchmark
condition. A preset is a *base*, so anything you set explicitly in `args` still wins:

~~~bash
# by condition name
curl -s localhost:8000/v1/evaluate -H 'content-type: application/json' -d '{
  "preset": "regional_turboprop",
  "x": [[0.8376, 0.0018, 0.6618, 0.0]],
  "m": 2
}'

# or by problem id -- resolves to that problem's condition
curl -s localhost:8000/v1/evaluate -H 'content-type: application/json' -d '{
  "preset": "ADO-M-2-2",
  "x": [[0.1935, 0.0, 0.1287, 0.0, 0.0002, 0.0006, 0.8274, 0.9026]], "m": 2
}'
~~~

| preset | Ma | Re_c | represents |
|---|---|---|---|
| `wind_tunnel` | 0.20 | 1e6 | 215 mm chord at 68.1 m/s, sea level |
| `regional_turboprop` | 0.40 | 1e7 | 1.93 m chord at 126.4 m/s at 20 000 ft (Saab 340B / ATR 42 class) |

**Driving the benchmark over HTTP.** The `/v1/benchmark` endpoints expose the twelve problems and their
released reference solutions, so an optimizer in any language can run the suite without importing
anything. They read `bench/data/manifest.json` in a repo checkout; from an installed package, point
`AIRDBM_BENCH_MANIFEST` at that file.

~~~bash
curl -s localhost:8000/v1/benchmark/problems | jq -r '.[].problem_id'
curl -s localhost:8000/v1/benchmark/problems/ADO-M-2-1 \
  | jq '{dimension, budget, history_schema, reference: (.reference | {n_front, hv_norm})}'

# evaluate at a problem's frozen condition -- no need to restate Ma, Re or the guard
curl -s localhost:8000/v1/benchmark/problems/ADO-M-2-1/evaluate \
  -H 'content-type: application/json' \
  -d '{"x": [[0.8376, 0.0018, 0.6618, 0.0]]}'
~~~

A minimal optimization loop is then: `GET` the problem for its dimension and budget, `POST` each
population to the evaluate endpoint, and stop at the budget. Compare on the same quantity the summaries
report — `frac_of_reference` for the single-objective problems, normalized hypervolume against
`hv_ref_norm` for the bi-objective ones.

Every evaluation is a real XFOIL solve, so requests are slow by nature (order of a second per
candidate, longer for pathological geometries). Submit a whole population in one call rather than
one candidate per request; the response reports `n_converged` and `wall_time_s`.

## Customizing Airfoil DbM

You can customize the AirDbM baseline set by providing your own airfoil coordinate files in Selig format (`.dat`).

1. Download or prepare the coordinate files in Selig format.
2. Place the files inside your airfoil database folder, such as `airfoilDB/`, or point `airfoil_db_dir` to a different folder.
3. Explicitly define `dbm_baselines` as an ordered list of airfoil names that matches the database files you want to use.

The order of `dbm_baselines` is important. The first `D` entries are used for a candidate with `D` design parameters, so the baseline ordering must match the intended morphing sequence.

Example:

~~~python
args = {
    'airfoil_db_dir': 'airfoilDB',
    'dbm_baselines': [
        'E195  (11.82%)',
        'FX 79-W-660A',
        'GOE 531 AIRFOIL',
        'EPPLER 864 STRUT AIRFOIL',
    ],
}
~~~

## Parallel Scaling Test

The parallel airfoil design and evaluation performance of `TestAirfoils` was measured by evaluating a fixed pool of design candidates across worker counts on a single node.

### Computing Environment

- Node type: Two 64-core AMD EPYC Milan processors @ 2.45 GHz (128 cores in total)
- Objective mode: `m=2` (multi-objective)
- Worker counts tested: `1` (serial), `2, 4, 8, 16, 32, 64, 128`

<!-- SCALING:BEGIN (regenerated by bench/report.py; do not edit by hand) -->

### Weak Scaling

Design candidates per worker: `24`. Every worker count draws its load as a nested prefix of one fixed i.i.d. design pool, so the workload composition is identical at every point and the expected time is flat.

| workers | candidates | time (sec) | sec/design | throughput (eval/sec) |
|---:|---:|---:|---:|---:|
| 1 | 24 | 311.950 | 12.998 | 0.08 |
| 2 | 48 | 318.825 | 13.284 | 0.15 |
| 4 | 96 | 324.722 | 13.530 | 0.30 |
| 8 | 192 | 319.867 | 13.328 | 0.60 |
| 16 | 384 | 313.963 | 13.082 | 1.22 |
| 32 | 768 | 316.052 | 13.169 | 2.43 |
| 64 | 1536 | 331.886 | 13.829 | 4.63 |
| 128 | 3072 | 411.335 | 17.139 | 7.47 |

### Strong Scaling

Total design candidates: `384` (the same set at every worker count).

| workers | candidates | time (sec) | speedup | efficiency |
|---:|---:|---:|---:|---:|
| 1 | 384 | 4921.202 | 1.00 | 1.000 |
| 2 | 384 | 2465.182 | 2.00 | 0.998 |
| 4 | 384 | 1240.874 | 3.97 | 0.991 |
| 8 | 384 | 626.653 | 7.85 | 0.982 |
| 16 | 384 | 320.502 | 15.35 | 0.960 |
| 32 | 384 | 165.232 | 29.78 | 0.931 |
| 64 | 384 | 91.000 | 54.08 | 0.845 |
| 128 | 384 | 60.819 | 80.92 | 0.632 |
<!-- SCALING:END -->

## Changelog

### v0.3.0

- **Released AirDbM-Bench, the optimization benchmark built on this evaluator.** Twelve frozen airfoil
  problems (single- and bi-objective, `D ∈ {4, 8, 12}`, two flight conditions) ship with reference
  solutions and the complete history of all 300 optimizer runs — 2.46 million XFOIL evaluations as plain
  gzipped CSV, documented in `bench/DATA.md`.
- **Added a JSON/HTTP service (`airdbm_service.py`)** so an optimizer in any language can drive the same
  evaluator, with named presets for the benchmark conditions and `/v1/benchmark` endpoints serving the
  twelve problems. Install with `pip install "airdbm[server]"` and run `airdbm-serve`.
- **Added a stability screen deciding which values may define a reference.** XFOIL can settle on more
  than one boundary layer solution for indistinguishable sections, so an isolated design may score far
  above its neighbours — reproducible but unreachable by search; such candidates are perturbation-tested
  before acceptance, also exposed as `airdbm_core.VerifyDesigns`.
- **Reference fronts are best-known, not proven optimal.** Each is pooled over every run of a problem and
  of every lower dimension, then improved by warm-started evaluations along the stall-margin range, so
  attainment is reported as a fraction of the best known.
- Documented parallel scaling on 128 cores: near-ideal through 32 workers, about `80.9x` speedup at 128.

### v0.2.1

- **Fixed the parallel `pickle data was truncated` failure with the full 12-baseline set.** Workers could
  read the cached airfoil database while another was still writing it; the cache is now written
  atomically and any partial cache is rebuilt from the raw `.dat` files instead of raising.
- **Guaranteed a non-negative stall margin (`delta_alpha`).** Non-negativity is now enforced at the
  objective boundary as well as the computation site, and sub-degree floating-point noise snaps to `0.0`.
- Exposed `airdbm_core.__version__` so an installed build can be identified at runtime.

### v0.2.0

- Baseline release: parallelized Design-by-Morphing airfoil generation with dynamic XFOIL evaluation
  (`Cl/Cd_max` and stall-margin objectives), Apptainer/native XFOIL backends, and multiprocessing across
  candidates.
