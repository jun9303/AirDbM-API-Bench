# AirDbM ADO benchmark

Twelve problems, `ADO-{S,M}-{2,4}-{1,2,3}`, spanning objective form (single/bi-objective), condition
(Ma 0.20 / Re_c 1e6 and Ma 0.40 / Re_c 1e7) and dimension (D = 4, 8, 12), with five seeds each. The budget
is `1024 x D` evaluations for every method, i.e. 4096 / 8192 / 12288 at D = 4 / 8 / 12.

| file | purpose |
|---|---|
| `problems.py` | problem specs, frozen evaluation arguments, objective helpers |
| `run.py` | run one (problem, method, seed) and write its evaluation history |
| `score.py` | rederive the reference fronts and the published attainment fractions |
| `gate.json` | the frozen reference set that arms the in-loop stability gate |
| `DATA.md` | layout and schema of everything under `data/` |
| `data/` | released evaluation histories, the two summaries, the problem manifest |

Reading the released data needs none of the above and no XFOIL: the histories are plain gzipped CSV, and
`DATA.md` documents the schema.

The scripts that post-process a campaign (references and statistics, figures and LaTeX tables, parallel
scaling, core-hour accounting, the end-of-study stability screen, reference-front refinement) are working
material rather than part of the benchmark, and are not published. The released surface is the problem
definitions, the runner, the scorer, the frozen gate constant and the data.

## Optimizers

Five per objective form. Every setting is the value its primary source specifies; where the implementing
library's default differs, the source wins. None of the seven population methods is implemented here:
they are run through pymoo, while Sobol comes from SciPy's QMC module and TPE from Optuna.
The released campaign used pymoo 0.6.2, Optuna 4.9.0 and cma 4.4.4 on Python 3.10.12, alongside
the NumPy, SciPy and Shapely versions pinned in `requirements.txt`.

| method | key settings | source |
|---|---|---|
| CMA-ES | `lambda = 4 + floor(3 ln D)`, `sigma0 = 0.3`, IPOP restarts | Hansen; cma via pymoo |
| DE | pop `10D`, `DE/rand/1/bin`, `F=0.8`, `CR=0.9` | Storn-Price classical settings |
| GA | pop 100, SBX(0.9, eta=20), PM(1/D, eta=20) | pymoo; Deb operators |
| PSO | swarm 25, `w=0.9`, `c1=c2=2.0`, adaptive | pymoo defaults; Kennedy-Eberhart |
| NSGA-II | pop 100, SBX(0.9, eta=20), PM(1/D, eta=20) | Deb et al. 2002 |
| SMS-EMOA | pop 100, least-hypervolume-contribution survival | Beume et al. 2007 |
| MOEA/D | 100 uniform weights, `T=20`, `delta=0.9` | Zhang-Li 2007 |
| Optuna TPE | multivariate TPE, ask/tell batch 8 | Optuna |
| Sobol | scrambled Sobol, batch 32 | SciPy |

Uniform random sampling is not carried as a separate baseline: preliminary runs, not part of this release,
showed no separation from scrambled Sobol, which is the better-distributed of the two.

No configuration is tuned to these problems, and a user tuning an optimizer against a specific problem
should therefore expect to beat these numbers.

## Reproduce

One run:

```bash
python bench/run.py --problem ADO-S-2-1 --method cmaes --seed 0 --max-workers 32
```

The full campaign is 300 runs, i.e. 12 problems x 5 optimizers x 5 seeds. Each is independent and they can
therefore be driven in any order, one process per run:

```bash
for prob in ADO-S-2-1 ADO-S-2-2 ADO-S-2-3 ADO-S-4-1 ADO-S-4-2 ADO-S-4-3; do
  for meth in cmaes de ga pso sobol; do
    for seed in 0 1 2 3 4; do
      python bench/run.py --problem "$prob" --method "$meth" --seed "$seed" --max-workers 32
    done
  done
done
```

Substitute the bi-objective problems (`ADO-M-...`) and their optimizers (`nsga2 smsemoa moead optuna sobol`)
for the other half. Set `--max-workers` to the optimizer's batch width, since a narrower batch leaves
additional workers idle: roughly 32 for GA, DE, NSGA-II and Sobol, 26 for PSO, 12 for CMA-ES, 8 for Optuna,
and 2 for the steady-state SMS-EMOA and MOEA/D, which evaluate one candidate at a time and therefore take
the longest.

`run.py` appends every evaluation to `<method>_seed<k>.partial.csv` as it goes, and thus an interrupted run
still leaves a usable history; the partial is removed once the final file is written. It will not replace
an existing history with a shorter one. Use `--seed 99` to smoke-test, or `--force` to overwrite
on purpose. Re-invoking with `--budget` equal to what a partial log already holds finalizes it on the
spot, touching no solver.

## Stability of a solution

XFOIL can settle on more than one boundary layer solution for two sections that are geometrically
near-identical, and an isolated design vector can therefore score far above every design around it. The
value is reproducible yet unattainable by search, and it is an artifact of the model rather than a property
of the shape. Two mechanisms rule such values out, and both are part of the problem definition.

**In-loop gate**, on by default and switched off with `run.py --gate off`. A candidate that the frozen
reference set does not dominate is perturbation-tested before its value is accepted, and a failure is
scored at the floor. The set is frozen in `gate.json`, which makes the verdict a function of the design
vector alone: it does not depend on what a run has already seen, on batch composition, or on later
regenerations of the summaries. Perturbation directions are likewise derived per design, from the bytes of
its own coordinates. Measured on the released campaign, the gate arms on 0.23% of bi-objective evaluations
and covers every design the end-of-study screen removed.

A frozen set cannot gate the campaign that produced it: the released runs were screened after the fact by
the end-of-study screen, and `gate.json` freezes that outcome so that every later study is gated against
one published constant. That file's own `rule` and `derived_from` fields record the arming condition and
how the set was built.

**End-of-study screen**, kept because it is what a study with no published front to gate against can do.
It perturbation-tests every candidate reference solution and scores at the floor any design whose
neighbors at radius `1e-6` differ from it by more than 2% in an objective. At the default
canonical-axes setting the check costs `1 + 2D` evaluations per candidate, i.e. 9, 17 and 25 at
`D` = 4, 8 and 12, and it applies only to the reference set; it therefore adds a fraction of a percent to
a campaign and nothing to an optimization run. The same check is available from the interface as
`airdbm_core.VerifyDesigns`.

## Reference solutions

A single-objective reference is the best value any released run of the problem attained, taken across
dimensions since the spaces are nested. A bi-objective reference front is the non-dominated set pooled over
every released run of the problem and of every lower dimension, together with the single-objective
reference designs of the same condition, re-evaluated bi-objectively because their own runs recorded no
stall margin.

That pool is then improved by additional evaluations spent on the front itself, outside the comparison: a
scalarized search fixes a stall-margin level and maximizes efficiency subject to it, warm-started from the
best design already known at that level, and thus it cannot return less than the pool already holds. Levels
are spread over the attainable margin range, with a few above the highest margin reached, which places the
extra evaluations where the front is sparse rather than where a crowding operator would leave them. These
evaluations are pooled into the reference and excluded from every per-method statistic.

References are therefore best-known rather than proven optimal: every point is an attainable design that
was evaluated, and further search can only add to the set. Report attainment as a fraction of the best
known.

The summaries in `data/` are derived from the histories beside them; a user who prefers to recompute
attainment from the raw CSVs can do so with `score.py` and should get the same numbers.
