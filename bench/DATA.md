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
    ...
  so_summary.json              single-objective references and per-method statistics
  mo_summary.json              bi-objective reference fronts and per-method statistics
  manifest.json                problem catalogue; also served by the HTTP API
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
| `running_hv` | dominated hypervolume of all rows up to and including this row, referenced to the origin |

The last column is derived from the ones before it and is stored only so a convergence curve needs no
recomputation. A failed evaluation scores the objective floor (`0`) rather than aborting its run, so the
row count always equals the budget.

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

## What is not here

The evaluations of the reference-front stage. What ships is the resulting front, in `mo_summary.json`,
together with the complete histories of every optimizer run; the additional scalarized-search evaluations
that improved each front beyond the pooled panel are working files and are not released. A front can
therefore be used and compared against, but not rederived from the raw files in this directory alone: the
panel part of it can be, the refinement part cannot.

Figures and LaTeX tables are not here either; they belong to the manuscript that presents them. Neither are
the auxiliary summaries used only for that write-up — cost accounting, parallel scaling, the stability
screen's exclusion list, optimum sparsity — nor the post-processing scripts that produce them, nor the
scheduler submission files, which are specific to one allocation and of no use elsewhere. The released
surface is deliberately just the histories, the two summaries and the manifest.

## Comparing your own optimizer

Run it against a problem's frozen evaluation arguments, at the same budget, and compare on the same
quantity the summaries report: `frac_of_reference` for the single-objective problems, or normalized
hypervolume against `hv_ref_norm` for the bi-objective ones. `bench/problems.py` holds the frozen
arguments and `bench/run.py` shows how a run is driven; the HTTP service exposes the same conditions as
named presets, so no Python is required.

Reference fronts are best-known rather than proven optimal. Each is the non-dominated set pooled over
every released run of the problem and of every lower dimension, improved by additional warm-started
evaluations; further search can only add to it. Treat `frac_of_reference_hv` as a fraction of the best
known, not of the true optimum.
