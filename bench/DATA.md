# AirDbM-Bench released data

Everything under `bench/data/` is the released benchmark: the complete per-evaluation history of every
optimizer run, plus two summary files and a problem manifest. No XFOIL and no part of this repository is
needed to read it — the histories are plain CSV.

```
bench/data/
  SO_D4_Re1e+06_Ma0.2/         one directory per problem
    cmaes_seed0.csv.gz         one file per (optimizer, seed)
    cmaes_seed1.csv.gz
    ...
  MO_D4_Re1e+06_Ma0.2/
    nsga2_seed0.csv.gz
    _refine.csv.gz             reference-stage evaluations: see "Reference-stage evaluations"
    _frontfill.csv.gz
    _trace_004.5.csv.gz
    _trace.grid.json           the frozen level grid those sweeps solved, and its provenance
    ...
  so_summary.json              single-objective references and per-method statistics
  mo_summary.json              bi-objective reference fronts and per-method statistics
  manifest.json                problem catalogue; also served by the HTTP API
  robustness.json              stability screen: parameters, verdicts and the exclusion list
```

## Problem identifiers and directory names

A problem is named `ADO-<O>-<C>-<N>`:

| field | meaning |
|---|---|
| `O` | objective form — `S` single-objective, `M` bi-objective |
| `C` | flight condition keyed by Mach — `2` is Ma 0.20 / Re_c 1e6, `4` is Ma 0.40 / Re_c 1e7 |
| `N` | dimension index — `1` is D 4, `2` is D 8, `3` is D 12 |

Directory names encode the same thing physically rather than by index: the objective form, the dimension,
then the flight condition, as `<SO|MO>_D<D>_Re<Re_c>_Ma<Ma>`.

| problem | D | objectives | budget | directory |
|---|---|---|---|---|
| ADO-S-2-1 | 4 | 1 | 4096 | `SO_D4_Re1e+06_Ma0.2` |
| ADO-S-2-2 | 8 | 1 | 8192 | `SO_D8_Re1e+06_Ma0.2` |
| ADO-S-2-3 | 12 | 1 | 12288 | `SO_D12_Re1e+06_Ma0.2` |
| ADO-S-4-1 | 4 | 1 | 4096 | `SO_D4_Re1e+07_Ma0.4` |
| ADO-S-4-2 | 8 | 1 | 8192 | `SO_D8_Re1e+07_Ma0.4` |
| ADO-S-4-3 | 12 | 1 | 12288 | `SO_D12_Re1e+07_Ma0.4` |
| ADO-M-2-1 | 4 | 2 | 4096 | `MO_D4_Re1e+06_Ma0.2` |
| ADO-M-2-2 | 8 | 2 | 8192 | `MO_D8_Re1e+06_Ma0.2` |
| ADO-M-2-3 | 12 | 2 | 12288 | `MO_D12_Re1e+06_Ma0.2` |
| ADO-M-4-1 | 4 | 2 | 4096 | `MO_D4_Re1e+07_Ma0.4` |
| ADO-M-4-2 | 8 | 2 | 8192 | `MO_D8_Re1e+07_Ma0.4` |
| ADO-M-4-3 | 12 | 2 | 12288 | `MO_D12_Re1e+07_Ma0.4` |

Every budget is `1024 x D`, so a comparison at fixed budget is a comparison at fixed budget-per-dimension.

## History files

`<method>_seed<k>.csv.gz`, one per optimizer and seed, five seeds each. Optimizers are `cmaes`, `de`,
`ga`, `pso`, `sobol` for the single-objective problems and `nsga2`, `smsemoa`, `moead`, `optuna`, `sobol`
for the bi-objective ones. Rows are in evaluation order, one row per objective evaluation, from the first
to the budget.

Single-objective columns:

```
x0 ... x{D-1}, y_cl_cd, running_best
```

Bi-objective columns:

```
x0 ... x{D-1}, y_cl_cd, y_dalpha, running_hv
```

| column | meaning |
|---|---|
| `x0 ... x{D-1}` | the design vector in `[0, 1]^D`, exactly as the optimizer proposed it |
| `y_cl_cd` | `(Cl/Cd)_max` over the angle-of-attack scan; `0` marks a non-converged evaluation |
| `y_dalpha` | stall margin in degrees, `alpha_stall - alpha*`, resolved to `0.1` |
| `running_best` | best `y_cl_cd` seen up to and including this row |
| `running_hv` | dominated hypervolume of all rows up to and including this row, in RAW objective units, referenced to the origin |

The last column is derived from the ones before it and is stored only so a convergence curve needs no
recomputation. A failed evaluation scores the objective floor (`0`) rather than aborting its run.

Three things about the zeros and the row counts are worth knowing before you filter anything.

**`y_cl_cd == 0` is a failure marker; `y_dalpha == 0` is not.** A converged evaluation can legitimately
return zero stall margin, meaning the airfoil reaches its lift peak at the same incidence at which it is
most efficient. There are 31,250 such rows across the bi-objective histories, against 1,662 rows where
both columns are zero. Masking on `y_dalpha == 0` therefore discards valid designs; mask on
`y_cl_cd == 0` instead.

**The histories are raw.** The stability screen described below is *not* applied to these rows. Its
verdicts and its exclusion list ship separately in `robustness.json`, so both views can be
reconstructed and neither is imposed on you.

**One run is one row short of its budget.** `SO_D4_Re1e+07_Ma0.4/cmaes_seed3.csv.gz` holds 4,095 rows
rather than 4,096, because CMA-ES met an internal convergence criterion and stopped just below the
allowance; the marker file beside it records this. Every other run has exactly `1024 x D` rows, and the
released campaign totals 2,457,599 rows rather than the nominal 2,457,600.

Two conventions matter when using the design vectors. The spaces are nested: a `D=4` vector is a valid
`D=8` vector once padded with zeros, and evaluates bit-identically. And the map from vector to weights is
invariant along rays through the origin, so the effective dimension is `D-1` and distinct vectors can
return identical objectives.

## Reading a history

```python
import gzip, numpy as np

with gzip.open("bench/data/MO_D4_Re1e+06_Ma0.2/nsga2_seed0.csv.gz", "rt") as f:
    a = np.loadtxt(f, delimiter=",", skiprows=1)

D = 4
X = a[:, :D]            # designs, in evaluation order
Y = a[:, D:D + 2]       # (Cl/Cd)_max and stall margin
curve = a[:, -1]        # running hypervolume
```

`np.loadtxt` reads `.csv.gz` directly by filename too, so the `gzip.open` is only needed if you want the
handle.

## Summaries

`so_summary.json` and `mo_summary.json` both have the shape `{"per_problem": {<problem id>: {...}}}`.

Single-objective entries carry `D`, `mach`, `reynolds`, `budget`, the released reference `y_ref` and the
design `x_ref` that attains it, and a `stats` list with one record per optimizer (mean, standard
deviation and best of the final value over seeds, and `frac_of_reference`).

Bi-objective entries carry the same problem fields plus `n_front`, the reference front as `front_Y`, its
absolute and normalized hypervolume `hv_ref_abs` and `hv_ref_norm`, three `representatives` — the
best-`Cl/Cd` corner, the best-margin corner and the compromise design closest to the ideal point — and a
`stats` list with per-optimizer normalized hypervolume and `frac_of_reference_hv`.

`manifest.json` is the problem catalogue: per problem the dimension, objective count, flight condition,
budget, seed count and reference solution. The HTTP service reads it to answer the
`/v1/benchmark/problems` endpoints, so it is part of the release rather than a derived summary.

## Reference-stage evaluations

A reference front is not just the best of the compared optimizers. After the panel finished, each front
was improved by an epsilon-constraint sweep: fix a stall-margin level, maximize efficiency at that
level, warm-started from the best design already known there. Those evaluations land where the panel
left the front sparse, including a few margin levels above anything the panel reached.

They ship in the `_`-prefixed files of each bi-objective problem directory — `_refine*`, `_frontfill*`
and the per-level `_trace*` sweeps — alongside `_trace.grid.json`, which records the frozen level grid
and its provenance. `mo_summary.json` labels the origin of every front point in `front_src`, where
`_refstage` marks a point that came from this stage and `_nested` one inherited from a lower dimension.

The share is large and worth stating plainly: on four of the six problems the reference stage supplies
91 to 97 per cent of the published front. **These evaluations are pooled into the reference only.** They
never enter a per-method statistic, because the refiner is not one of the compared optimizers; it is
part of how the target is built.

## Stability screen

`robustness.json` holds the screen's parameters, its per-candidate verdicts, and the values it excluded.
An objective value that no neighbour of its design reproduces under a small perturbation is isolated:
search cannot reach it and it cannot serve as a reference, so it is scored at the failure floor when the
summaries are built. In v0.3.0 exactly one value is excluded, on `ADO-M-2-3`.

The histories themselves are untouched, so the screen is a view you can apply or ignore.
`bench/score.py` applies it, which is why its rederived fronts match the published ones exactly.

## Reproducing the published numbers

`bench/score.py` recomputes everything the summaries report, from the files in this directory. It needs
numpy and nothing else — no XFOIL, no pymoo, no part of the API.

```
python bench/score.py
```

This rederives all six reference fronts and rechecks all sixty per-method attainment fractions. Import
it to score your own optimizer on the same footing:

```python
from bench.score import score_mo_run, score_so_run, norm_hv, reference_front
```

Two hypervolumes appear in this release and they are not interchangeable. The `running_hv` column and
`hv_ref_abs` are in raw objective units referenced to the origin. The scored quantity, `hv_ref_norm` and
every `frac_of_reference_hv`, divides each objective by the reference front's own maximum and puts the
reference point at `REF_OFFSET = 0.05` below the origin of those normalized objectives, that is, five
per cent worse than the floor a failed evaluation scores. Normalizing matters because efficiency runs to
a few hundred while margin runs to a few tens of degrees, so an origin-referenced hypervolume on raw
values would fix an arbitrary exchange rate between one unit of efficiency and one degree of margin.

Both reported fractions are **means over the five seeds**, not best-of-seed.

## What is not here

Figures and LaTeX tables belong to the manuscript that presents them. Neither they nor the auxiliary
summaries used only for that write-up — cost accounting, parallel scaling, optimum sparsity — are here,
nor the post-processing scripts that produce them, nor the scheduler submission files, which are
specific to one allocation and of no use elsewhere.

## Comparing your own optimizer

Run it against a problem's frozen evaluation arguments, at the same budget, and compare on the same
quantity the summaries report: `frac_of_reference` for the single-objective problems, or normalized
hypervolume against `hv_ref_norm` for the bi-objective ones. `bench/problems.py` holds the frozen
arguments and `bench/run.py` shows how a run is driven; the HTTP service exposes the same conditions as
named presets, so no Python is required.

Reference fronts are best-known rather than proven optimal. Each is the non-dominated set pooled over
every released run of the problem, the front already established at the next lower dimension, and the
reference-stage evaluations; further search can only add to it. Treat `frac_of_reference_hv` as a
fraction of the best known, not of the true optimum.

If you reimplement the stability gate, load `gate.json` rather than `mo_summary.json`. For the
single-objective problems the two agree, but the gate's bi-objective fronts deliberately freeze the
panel pool as it stood *before* the reference stage, and are the weaker of the two sets. Gating against
the published front instead would arm far more rarely and would tie the problem definition to a target
that later search can still improve.
