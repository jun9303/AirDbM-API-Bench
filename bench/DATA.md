# AirDbM-Bench released data

Everything under `bench/data/` is the released benchmark: the complete per-evaluation history of every
optimizer run, the reference-stage evaluations behind every target, two summary files, a problem
manifest, and the stability and cost records. Reading it needs neither XFOIL nor any part of this
repository, since the histories are plain CSV.

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
  cost.json                    core-hours per run, and the campaign total
```

## Problem identifiers and directory names

A problem is named `ADO-<O>-<C>-<N>`, where `O` is the objective form (`S` single-objective, `M`
bi-objective), `C` the flight condition keyed by Mach (`2` is Ma 0.20 / Re_c 1e6, `4` is Ma 0.40 /
Re_c 1e7), and `N` the dimension index (`1` is D 4, `2` is D 8, `3` is D 12). Directory names encode the
same information physically rather than by index, as `<SO|MO>_D<D>_Re<Re_c>_Ma<Ma>`.

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

Every budget is `1024 x D`; a comparison at fixed budget is therefore a comparison at fixed
budget-per-dimension.

## History files

`<method>_seed<k>.csv.gz`, one per optimizer and seed, five seeds each. Optimizers are `cmaes`, `de`,
`ga`, `pso`, `sobol` for the single-objective problems and `nsga2`, `smsemoa`, `moead`, `optuna`, `sobol`
for the bi-objective ones. Rows are in evaluation order, one row per objective evaluation, from the first
to the budget. The columns are `x0 ... x{D-1}, y_cl_cd, running_best` for a single-objective problem and
`x0 ... x{D-1}, y_cl_cd, y_dalpha, running_hv` for a bi-objective one:

| column | meaning |
|---|---|
| `x0 ... x{D-1}` | the design vector in `[0, 1]^D`, exactly as the optimizer proposed it |
| `y_cl_cd` | `(Cl/Cd)_max` over the angle-of-attack scan; `0` marks a non-converged evaluation |
| `y_dalpha` | stall margin in degrees, `alpha_stall - alpha*`, resolved to `0.1` |
| `running_best` | best `y_cl_cd` seen up to and including this row |
| `running_hv` | dominated hypervolume of all rows up to and including this row, in RAW objective units, referenced to the origin |

The last column is derived from the ones before it and is stored only so that a convergence curve needs
no recomputation. A failed evaluation scores the objective floor (`0`) rather than aborting its run.

```python
import numpy as np
a = np.loadtxt("bench/data/MO_D4_Re1e+06_Ma0.2/nsga2_seed0.csv.gz", delimiter=",", skiprows=1)
D = 4
X, Y, curve = a[:, :D], a[:, D:D + 2], a[:, -1]   # designs, objectives, running hypervolume
```

Four points are worth knowing before you filter anything.

**`y_cl_cd == 0` is a failure marker, whereas `y_dalpha == 0` is not.** A converged evaluation can
validly return zero stall margin, meaning the airfoil reaches its lift peak at the same incidence at
which it is most efficient. There are 31,250 such rows across the bi-objective histories, against 1,662
rows where both columns are zero. Masking on `y_dalpha == 0` therefore discards valid designs; mask on
`y_cl_cd == 0` instead.

**The histories are raw.** The stability screen described below is *not* applied to these rows. Its
verdicts and its exclusion list ship separately in `robustness.json`, and thus both views can be
reconstructed and neither is imposed on you.

**One run is one row short of its budget.** `SO_D4_Re1e+07_Ma0.4/cmaes_seed3.csv.gz` holds 4,095 rows
rather than 4,096, because CMA-ES met an internal convergence criterion and stopped just below the
allowance; the sidecar `cmaes_seed3.short.json` next to it records the count and the reason. Every other run has exactly `1024 x D` rows, and the
released campaign therefore totals 2,457,599 rows rather than the nominal 2,457,600.

**Some methods re-evaluate designs they have already seen, and the budget is charged for it.**
The evaluator is a deterministic function of the design vector, so a repeated vector returns the values
recorded the first time and buys no new information. The rates differ sharply by method. Across the
released runs, MOEA/D repeats 37.5% of its 245,760 charged evaluations, from 30.8% on `ADO-M-2-1` up to
44.0% on `ADO-M-2-3`; particle swarm repeats 1.6%, differential evolution 0.03%, and CMA-ES, the genetic
algorithm, NSGA-II, SMS-EMOA, Optuna TPE and Sobol repeat none. The reason is configuration, not a data
fault: the genetic and NSGA-II runs enable pymoo's duplicate elimination and the MOEA/D run does not.
Read the MOEA/D curves with that in mind, since they cover fewer distinct designs than their budget
implies. Pooling the runs of each problem, 104,593 of the 2,457,599 rows repeat a design vector that appeared
earlier, and every one of them returns the objective values of its first appearance.

**Methods that share a population size share their seeded initial sample.** At a given seed, MOEA/D and
NSGA-II open on the same 100 designs, and differential evolution and the genetic algorithm on the same
40, 80 or 100 designs at `D = 4, 8, 12`. This follows from seeding one sampler and is why those runs
overlap at the start.

**The design spaces are nested and ray-invariant.** A `D=4` vector is a valid `D=8` vector once padded
with zeros, and evaluates bit-identically. The map from vector to weights is invariant along rays through
the origin, which leaves the effective dimension at `D-1` and allows distinct vectors to return identical
objectives.

## Where the baseline geometries come from

The twelve DbM baselines in `airfoilDB/` are taken from the UIUC Airfoil Coordinates Database
(<https://m-selig.ae.illinois.edu/ads/coord_database.html>). They are plain text in the Selig
convention that XFOIL reads: a name line, then one `x y` pair per line tracing a single contour from the
trailing edge along the upper surface to the leading edge and back along the lower surface. Line endings
are CRLF and numeric precision varies between files, both inherited from the source.

Six files are the UIUC originals with no change at all: `e195.dat`, `fx79w660a.dat`,
`griffith30SymSuction.dat`, `s9104.dat`, `ah93w480b.dat`, `ah81k144wfKlappe.dat`. (`s9104.dat` has a
`# Airfoil by Michael Selig / CC BY 4.0` line and eighteen-decimal coordinates because the UIUC file
does.)

UIUC publishes the other six in the two-surface Lednicer layout, which opens with a point count line and
then lists each surface outward from the leading edge. XFOIL wants one contour, so `goe531.dat`,
`e864.dat`, `r1145msm.dat`, `chen.dat`, `e664ex.dat` and `saratov.dat` were rewritten in Selig order:
reverse the upper block, join the two, drop the leading-edge point that the source lists twice. No
coordinate value was altered and no point was added, removed or resampled, so undoing the ordering
returns the source coordinates. This is the only preparation applied to any input geometry.

## Summaries

`so_summary.json` and `mo_summary.json` both have the shape `{"per_problem": {<problem id>: {...}}}`.

Single-objective entries carry `D`, `mach`, `reynolds`, `budget`, the released reference `y_ref` and the
design `x_ref` that attains it, and a `stats` list with one record per optimizer: `final_mean`,
`final_std` and `final_best` of the final value over seeds, `n_seed`, and `frac_of_reference`.

Bi-objective entries carry the same problem fields plus `n_front`, the reference front as `front_Y`, its
absolute and normalized hypervolume `hv_ref_abs` and `hv_ref_norm`, three `representatives` (the
best-`Cl/Cd` corner, the best-margin corner and the compromise design closest to the ideal point) and a
`stats` list with `final_hv_norm_mean`, `final_hv_norm_median`, `final_hv_norm_std`, `median_ci95`,
`n_seeds` and `frac_of_reference_hv`.

Note that the two `std` fields use different estimators: `final_std` is the population standard deviation
(`ddof=0`) and `final_hv_norm_std` the sample standard deviation (`ddof=1`), which differ by a factor of
`sqrt(5/4)` over five seeds. Both reported attainment fractions are **means over the five seeds** rather
than best-of-seed.

`manifest.json` is the problem catalogue: per problem the dimension, objective count, flight condition,
budget, seed count and reference solution. The HTTP service reads it to answer the
`/v1/benchmark/problems` endpoints, and it is therefore part of the release rather than a derived summary.

## Reference-stage evaluations

A reference front is more than the best of the compared optimizers. After the panel finished, each front
was improved by an epsilon-constraint sweep: fix a stall-margin level, then maximize efficiency at that
level, warm-started from the best design already known there. Those evaluations land where the panel left
the front sparse, including a few margin levels above anything the panel reached.

They ship in the `_`-prefixed files of each bi-objective problem directory (`_refine*`, `_frontfill*` and
the per-level `_trace*` sweeps) alongside `_trace.grid.json`, which records the frozen level grid and its
provenance. `mo_summary.json` labels the origin of every front point in `front_src`, where `_refstage`
marks a point contributed by this stage and `_nested` one inherited from a lower dimension.

The share is large and we state it: on four of the six problems the reference stage supplies
91 to 97 per cent of the published front. **These evaluations are pooled into the reference only.** They
never enter a per-method statistic, because the refiner is not one of the compared optimizers; it is part
of how the target is built.

## Stability screen

`robustness.json` holds the screen's parameters, its per-candidate verdicts, and the values it excluded.
An objective value that no neighbour of its design reproduces under a small perturbation is isolated:
search cannot reach it and it cannot serve as a reference, and it is therefore scored at the failure floor
when the summaries are built. In this release exactly one value is excluded, on `ADO-M-2-3`. The
histories themselves are untouched, which leaves the screen a view you can apply or ignore.
`bench/score.py` applies it, which is why its rederived fronts match the published ones exactly.

If you reimplement the in-loop gate, load `gate.json` rather than `mo_summary.json`. For the
single-objective problems the two agree; the gate's bi-objective fronts intentionally freeze the panel pool
as it stood *before* the reference stage, and are the weaker of the two sets. Gating against the published
front instead would arm far more rarely and would tie the problem definition to a target that later search
can still improve.

## Cost

`cost.json` records the core-hours each of the 300 runs consumed and the campaign total, 15,705
core-hours over 2,457,599 evaluations. Dividing one by the other gives a campaign mean of about 23 s per
evaluation on one core, which is a quotient rather than a calibrated timing of a single design. That
figure is the reason the suite is posed against XFOIL: slow enough that an optimizer has to spend its
budget carefully, and affordable enough that a campaign of this size can be repeated.

## Reproducing the published numbers

`bench/score.py` rederives all six reference fronts and rechecks all sixty per-method attainment
fractions from the files in this directory. It needs numpy alone, with no XFOIL, no pymoo and no part of
the API:

```
python bench/score.py
```

On the released files every one of the sixty fractions agrees with the published value to within
2.2e-16, which is at the level of floating-point rounding.

Import it to score your own optimizer on the same footing:

```python
from bench.score import score_mo_run, score_so_run, norm_hv, reference_front
```

Two hypervolumes appear in this release and they are not interchangeable. The `running_hv` column and
`hv_ref_abs` are in raw objective units referenced to the origin. The scored quantity, `hv_ref_norm` and
every `frac_of_reference_hv`, divides each objective by the reference front's own maximum and places the
reference point at `REF_OFFSET = 0.05` below the origin of those normalized objectives, i.e. five per cent
worse than the floor a failed evaluation scores. Normalizing matters because efficiency runs to a few
hundred while margin runs to a few tens of degrees; an origin-referenced hypervolume on raw values would
otherwise fix an arbitrary exchange rate between one unit of efficiency and one degree of margin.

## Comparing your own optimizer

Run it against a problem's frozen evaluation arguments, at the same budget, and compare on the same
quantity the summaries report: `frac_of_reference` for the single-objective problems, or normalized
hypervolume against `hv_ref_norm` for the bi-objective ones. `bench/problems.py` holds the frozen
arguments and `bench/run.py` shows how a run is driven; the HTTP service exposes the same conditions as
named presets, and thus no Python is required.

Reference fronts are best-known rather than proven optimal. Each is the non-dominated set pooled over
every released run of the problem, the front already established at the next lower dimension, and the
reference-stage evaluations; further search can only add to it. Treat `frac_of_reference_hv` as a fraction
of the best known rather than of the true optimum.

## What is not here

Figures and LaTeX tables belong to the manuscript that presents them. Neither they nor the auxiliary
summaries used only for that write-up (cost accounting, parallel scaling, optimum sparsity) are here, nor
the post-processing scripts that produce them.
