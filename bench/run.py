import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problems import (DATA, MO_METHODS, SO_METHODS, SMOKE_SEED, budget_for, evaluate,
                      gate_set, hypervolume, non_dominated, running_best, running_hv, screen,
                      spec)

BATCH = 32
# Population override for reference-building runs. The released panel runs at the published
# defaults (POP["n"] = 0); a reference run sets it, because front density is capped by the size of
# the surviving population, not by the budget: pooling thirty gated runs at pop 100 yields a front
# of 36-78 points, no larger than the best single run.
POP = {"n": 0}
# Early termination for reference-building runs. The panel is compared at a fixed budget, so it must
# never stop early; a run searching for the true optimum has no budget to respect and should stop when
# it stops learning instead. STOP["tol"] is the relative gain over the last STOP["patience"]
# generations below which the search is declared converged -- hypervolume for m=2, best-so-far for
# m=1 -- and it is not checked before STOP["min_gen"] generations have passed. tol = 0 disables it.
STOP = {"tol": 0.0, "patience": 20, "min_gen": 100, "curve": [], "reason": None,
        "floor": -float("inf")}
FLOOR_RTOL = 1e-9      # slack for the serialized precision of the frozen reference constant
TRACE = {"Y": []}
# Stability gate: dominance against the frozen reference set of gate.json, on by default and
# switched off with --gate off. GATE["n"] counts the candidates floored during a run and is
# reported in the run's summary line.
GATE = {"on": True, "n": 0}
PARTIAL = {"fh": None, "path": None}
CKPT = {"path": None}
# Every pymoo-driven method checkpoints, so no run can be lost to a walltime kill regardless of
# how its runtime turns out. The cost is one dill dump per generation. Sobol is a deterministic
# sequence and Optuna keeps its own study state, so neither goes through this path.
CKPT_METHODS = ("cmaes", "de", "ga", "pso", "nsga2", "smsemoa", "moead")


def _open_partial(s, method, seed, mode):
    """Open the append-only evaluation log. `mode` is 'append' when continuing an interrupted run
    and 'fresh' when starting one; appending to a truncated log from a *different* search
    trajectory would splice two runs into one history."""
    d = DATA / s["tag"]
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{method}_seed{seed}.partial.csv"
    PARTIAL["path"] = f
    write_header = (mode == "fresh") or not f.exists()
    fh = open(f, "w" if mode == "fresh" else "a", buffering=1)
    if write_header:
        fh.write(",".join([f"x{i}" for i in range(s["D"])] +
                          (["y_cl_cd"] if s["m"] == 1 else ["y_cl_cd", "y_dalpha"])) + "\n")
    PARTIAL["fh"] = fh


def _log_partial(X, Y):
    f = PARTIAL["fh"]
    if f is None:
        return
    X = np.atleast_2d(X)
    Y = np.atleast_2d(np.asarray(Y, dtype=float).reshape(len(X), -1))
    for xr, yr in zip(X, Y):
        f.write(",".join(f"{v:.18e}" for v in list(xr) + list(yr)) + "\n")


def _ev(X, s, max_workers=None):
    """Every optimizer's evaluations funnel through here, so the stability gate applies uniformly.

    The gate is a published constant (problems.gate_set), so an optimizer proposing a record-breaking
    design is screened identically whenever and however it proposes it -- the objective stays a
    function of the design vector alone. The screened value is what gets logged, so the released
    history is the history the optimizer actually saw.
    """
    Y = evaluate(X, s, max_workers=max_workers)
    if GATE["on"]:
        Y, floored = screen(X, Y, s, max_workers=max_workers, on=GATE["on"])
        GATE["n"] += len(floored)
    _log_partial(X, Y)
    if STOP["tol"] > 0:
        TRACE["Y"].append(np.atleast_2d(np.asarray(Y, dtype=float)).reshape(len(np.atleast_2d(X)), -1))
    return Y


def _problem(s, mw):
    from pymoo.core.problem import Problem

    class P(Problem):
        def __init__(self):
            super().__init__(n_var=s["D"], n_obj=s["m"], n_ieq_constr=0,
                             xl=np.zeros(s["D"]), xu=np.ones(s["D"]))

        def _evaluate(self, X, out, *a, **k):
            Y = _ev(X, s, max_workers=mw)
            out["F"] = -np.asarray(Y, dtype=float).reshape(-1, s["m"])

    return P()


def _floor(s):
    """The value a reference run must reach before it may call itself finished.

    A run below the frozen reference has contributed nothing to the target, so a plateau there is not
    convergence -- it is a run stuck in a worse basin, and stopping it wastes the allowance it was
    given. The threshold is the published constant of gate.json, not anything this run has seen.
    """
    kind, ref = gate_set(s["pid"]) if "pid" in s else (None, None)
    if kind == "best":
        return float(ref)
    if kind == "front":
        return float(hypervolume(non_dominated(np.asarray(ref, dtype=float))))
    return -np.inf


def _converged(s):
    """True once the accumulated search has stopped improving.

    The metric is the quantity the benchmark actually reports: origin-referenced hypervolume of the
    accumulated non-dominated set for m = 2, best-so-far objective for m = 1. It is computed over the
    whole history rather than the current population, because that is what enters the reference.
    """
    if STOP["tol"] <= 0 or not TRACE["Y"]:
        return False
    Y = np.vstack(TRACE["Y"])
    v = float(Y[:, 0].max()) if s["m"] == 1 else float(hypervolume(non_dominated(Y[:, :2])))
    STOP["curve"].append(v)
    c = STOP["curve"]
    if len(c) < max(STOP["min_gen"], STOP["patience"] + 1):
        return False
    # The floor is a decimal constant serialized into gate.json, so a run that rediscovers the
    # reference optimum exactly still lands a few parts in 1e11 below the stored value. A strict
    # comparison would make the floor unreachable precisely in the case it is meant to admit -- a run
    # that has matched the reference -- so it is applied with a relative tolerance well inside the
    # constant's twelve significant digits.
    if v < STOP["floor"] - FLOOR_RTOL * abs(STOP["floor"]):
        return False
    prev = c[-1 - STOP["patience"]]
    if prev <= 0:
        return False
    gain = (v - prev) / abs(prev)
    if gain < STOP["tol"]:
        STOP["reason"] = (f"converged: {100 * gain:.4f}% gain over the last {STOP['patience']} "
                          f"generations (< {100 * STOP['tol']:.4f}%) after {len(c)} generations, "
                          f"{len(Y)} evaluations")
        return True
    return False


def _pymoo(s, algo, budget, seed, mw):
    """Run a pymoo algorithm, checkpointing after every generation.

    The search state is serialized with dill (plain pickle cannot handle pymoo's decorator
    closures) so a run killed at the walltime limit resumes in a follow-on job
    instead of being lost; the evaluation history itself lives in the append-only partial log. The
    problem instance closes over the evaluator, so it is detached around the dump and rebound
    afterwards.
    """
    problem = _problem(s, mw)
    ck = CKPT["path"]
    if ck is not None and ck.exists():
        import dill
        algo = dill.load(open(ck, "rb"))
        algo.problem = problem
        # The checkpoint carries the termination it was created with, and `has_terminated` reads a
        # cached percentage, so a run resumed under a RAISED budget would exit immediately. Rebind a
        # fresh criterion to the current budget; extending a reference run is the intended use.
        from pymoo.termination.max_eval import MaximumFunctionCallTermination
        algo.termination = MaximumFunctionCallTermination(budget)
        print(f"resumed from checkpoint at n_eval={algo.evaluator.n_eval}, "
              f"continuing to {budget}", flush=True)
    else:
        algo.setup(problem, termination=("n_evals", budget), seed=seed, verbose=False)
    while algo.has_next():
        algo.next()
        if _converged(s):
            print(f"  early termination -- {STOP['reason']}", flush=True)
            # Record it before finalizing. Everything after this point (rebuilding the derived
            # column, writing the history) is bookkeeping that a walltime kill can interrupt, and
            # the fact that this run stopped because it converged -- rather than because it ran out
            # of allowance -- is not recoverable from the history afterwards.
            if ck is not None:
                ck.with_suffix(".converged.json").write_text(json.dumps(
                    {"reason": STOP["reason"], "n_generations": len(STOP["curve"]),
                     "floor": STOP["floor"], "curve": STOP["curve"]}, indent=2) + "\n")
            break
        # pymoo's loopwise algorithms (MOEA/D) suspend a coroutine in `algo.generator` to hold
        # position *within* a generation; that object cannot be serialized, and discarding it would
        # silently restart the generation. So checkpoint only at a generation boundary, where the
        # coroutine has already been retired. At most one generation of work is lost on a kill.
        if ck is not None and getattr(algo, "generator", None) is None:
            p = algo.problem
            algo.problem = None
            tmp = ck.with_suffix(".tmp")
            import dill
            with open(tmp, "wb") as fh:
                dill.dump(algo, fh)
            os.replace(tmp, ck)
            algo.problem = p
    return None, None


def m_sobol(s, budget, seed, mw):
    from scipy.stats import qmc
    X = qmc.Sobol(d=s["D"], scramble=True, seed=seed).random(budget)
    Xs, Ys = [], []
    for i in range(0, budget, BATCH):
        Xb = X[i:i + BATCH]
        Xs.append(Xb)
        Ys.append(np.asarray(_ev(Xb, s, max_workers=mw)).reshape(len(Xb), -1))
    return np.vstack(Xs), np.vstack(Ys)


# Operator settings follow the primary sources, not the library defaults, wherever the two differ.
# Deb et al. (2002) p.187: SBX with p_c = 0.9 and eta_c = 20; polynomial mutation with eta_m = 20
# and p_m = 1/n. pymoo's SBX default is eta = 15, so it is overridden here.
def _sbx_pm(D):
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    return SBX(prob=0.9, eta=20), PM(prob=1.0 / D, eta=20)


def m_cmaes(s, budget, seed, mw):
    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
    # lambda = 4 + floor(3 ln D) is the cma/Hansen default; sigma0 = 0.3 of the variable range is
    # the value Hansen recommends for a bounded box (pymoo's generic default of 0.1 is not used).
    # restarts with population doubling is the library's IPOP scheme, for the multimodal D = 12 case.
    return _pymoo(s, CMAES(sigma=0.3, restarts=2, incpopsize=2), budget, seed, mw)


def m_de(s, budget, seed, mw):
    from pymoo.algorithms.soo.nonconvex.de import DE
    # The classical Storn-Price settings that pymoo's own documentation cites: population 10*D,
    # DE/rand/1/bin, F = 0.8, CR = 0.9. pymoo's code default (100, DE/best/1/bin) is not used.
    return _pymoo(s, DE(pop_size=POP["n"] or 10 * s["D"], variant="DE/rand/1/bin", F=0.8, CR=0.9),
                  budget, seed, mw)


def m_ga(s, budget, seed, mw):
    from pymoo.algorithms.soo.nonconvex.ga import GA
    cx, mu = _sbx_pm(s["D"])
    return _pymoo(s, GA(pop_size=POP["n"] or 100, crossover=cx, mutation=mu, eliminate_duplicates=True),
                  budget, seed, mw)


def m_pso(s, budget, seed, mw):
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    # pymoo defaults throughout: swarm 25, w = 0.9, c1 = c2 = 2.0, adaptive.
    return _pymoo(s, PSO(), budget, seed, mw)


def m_nsga2(s, budget, seed, mw):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    cx, mu = _sbx_pm(s["D"])
    return _pymoo(s, NSGA2(pop_size=POP["n"] or 100, crossover=cx, mutation=mu, eliminate_duplicates=True),
                  budget, seed, mw)


def m_smsemoa(s, budget, seed, mw):
    from pymoo.algorithms.moo.sms import SMSEMOA
    cx, mu = _sbx_pm(s["D"])
    # Beume et al. (2007): steady-state (mu+1), survival by least hypervolume contribution -- the
    # criterion a hypervolume-based Bayesian acquisition maximizes in expectation, but computed
    # directly, so cost per evaluation does not grow with the observation count. One offspring per
    # generation serializes the budget, which the per-generation checkpoint makes safe: a job
    # killed at the walltime ceiling resumes from its last generation.
    return _pymoo(s, SMSEMOA(pop_size=POP["n"] or 100, n_offsprings=1, crossover=cx, mutation=mu),
                  budget, seed, mw)


def m_moead(s, budget, seed, mw):
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.util.ref_dirs import get_reference_directions
    cx, mu = _sbx_pm(s["D"])
    # Zhang and Li (2007) two-objective setting: 100 weight vectors, T = 20, delta = 0.9. The
    # subproblem-by-subproblem update is inherently serial; checkpointing covers the walltime.
    ref = get_reference_directions("uniform", 2, n_partitions=(POP["n"] or 100) - 1)
    return _pymoo(s, MOEAD(ref, n_neighbors=20, prob_neighbor_mating=0.9,
                           crossover=cx, mutation=mu), budget, seed, mw)


def m_optuna(s, budget, seed, mw, batch=8):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(directions=["maximize", "maximize"],
                                sampler=optuna.samplers.TPESampler(multivariate=True, seed=seed))
    Xs, Ys, rem = [], [], budget
    while rem > 0:
        b = min(batch, rem)
        tr = [study.ask() for _ in range(b)]
        X = np.array([[t.suggest_float(f"x{i}", 0.0, 1.0) for i in range(s["D"])] for t in tr])
        Y = np.asarray(_ev(X, s, max_workers=mw)).reshape(b, -1)
        for t, y in zip(tr, Y):
            study.tell(t, [float(y[0]), float(y[1])])
        Xs.append(X)
        Ys.append(Y)
        rem -= b
    return np.vstack(Xs), np.vstack(Ys)


METHODS = {"cmaes": m_cmaes, "de": m_de, "ga": m_ga, "pso": m_pso, "sobol": m_sobol,
           "nsga2": m_nsga2, "smsemoa": m_smsemoa, "moead": m_moead, "optuna": m_optuna}


def save(s, method, seed, X, Y, budget, force):
    if X is None:
        d = np.atleast_2d(np.loadtxt(PARTIAL["path"], delimiter=",", skiprows=1))
        X, Y = d[:, :s["D"]], d[:, s["D"]:]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(np.asarray(Y, dtype=float)).reshape(len(X), -1)
    X, Y = X[:budget], Y[:budget]
    out = DATA / s["tag"]
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{method}_seed{seed}.csv"
    if target.exists() and not force:
        n = sum(1 for _ in open(target)) - 1
        if n > len(Y):
            raise SystemExit(f"{target} has {n} rows; refusing to replace with {len(Y)}. "
                             f"Use --seed {SMOKE_SEED} to smoke-test or --force to overwrite.")
    if s["m"] == 1:
        y = Y[:, 0]
        cols = np.column_stack([X, y, running_best(y)])
        head = ",".join([f"x{i}" for i in range(X.shape[1])] + ["y_cl_cd", "running_best"])
        best = {"best_cl_cd": float(y.max())}
    else:
        cols = np.column_stack([X, Y[:, :2], running_hv(Y[:, :2])])
        head = ",".join([f"x{i}" for i in range(X.shape[1])] + ["y_cl_cd", "y_dalpha", "running_hv"])
        best = {"final_hv": float(running_hv(Y[:, :2])[-1]),
                "n_nondominated": int(len(non_dominated(Y[:, :2])))}
    np.savetxt(target, cols, delimiter=",", header=head, comments="")
    return {"problem": s["pid"], "tag": s["tag"], "method": method, "seed": int(seed),
            "n_eval": int(len(Y)), "budget": int(budget),
            "budget_shortfall": int(budget - len(Y)),
            # which gate the run was scored under, and how often it floored a candidate: a released
            # run states its own protocol rather than relying on the reader to know it
            "gate": ("on" if GATE["on"] else "off"), "gate_events": int(GATE["n"]), **best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--pop", type=int, default=0,
                    help="population override (GA/DE/NSGA-II/SMS-EMOA/MOEA-D). Only for "
                         "reference-building runs; the compared panel uses the published defaults")
    ap.add_argument("--label", default="",
                    help="mark this run as reference-building: output goes to _<method><label>_"
                         "seed<k>.csv, which report.py pools into the reference front and excludes "
                         "from every per-method statistic")
    ap.add_argument("--stop-tol", type=float, default=0.0,
                    help="relative gain over --stop-patience generations below which a run is "
                         "declared converged and terminated early (hypervolume for m=2, best-so-far "
                         "for m=1). 0 disables. Intended for reference-building runs only: the "
                         "compared panel must always spend its full published budget")
    ap.add_argument("--stop-patience", type=int, default=20, help="generations in the window")
    ap.add_argument("--stop-min-gen", type=int, default=100,
                    help="generations that must pass before the test is applied")
    ap.add_argument("--gate", default="on", choices=("on", "off"),
                    help="perturbation-screen any candidate the frozen reference set of gate.json "
                         "does not dominate (default), or switch the screen off")
    a = ap.parse_args()
    GATE["on"] = (a.gate == "on")
    POP["n"] = a.pop
    STOP.update(tol=a.stop_tol, patience=a.stop_patience, min_gen=a.stop_min_gen)
    if a.stop_tol > 0 and not a.label:
        raise SystemExit("--stop-tol is for reference-building runs; use --label so the run is kept "
                         "out of the optimizer comparison, whose budget must be fixed")
    if a.label and not re.fullmatch(r"[a-z0-9]+", a.label):
        raise SystemExit("--label must be lowercase alphanumeric (it becomes part of a filename "
                         "report.py parses)")
    # The run's identity in the filesystem. A labelled run must not collide with the released
    # history of the same method and seed, partials and checkpoints included.
    name = f"_{a.method}{a.label}" if a.label else a.method

    s = spec(a.problem)
    allowed = SO_METHODS if s["m"] == 1 else MO_METHODS
    if a.method not in allowed:
        raise SystemExit(f"{a.method} is not a {s['m']}-objective method; use one of {allowed}")
    budget = a.budget if a.budget > 0 else budget_for(s, a.method)
    if a.stop_tol > 0:
        STOP["floor"] = _floor(s)
        print(f"convergence floor (frozen reference): {STOP['floor']:.4f}", flush=True)

    part = DATA / s["tag"] / f"{name}_seed{a.seed}.partial.csv"
    ckpt = DATA / s["tag"] / f"{name}_seed{a.seed}.ckpt"
    have = (sum(1 for _ in open(part)) - 1) if part.exists() else 0
    resumable = (not a.no_checkpoint) and a.method in CKPT_METHODS and ckpt.exists()

    if have >= budget:
        # Idempotent: the log already holds the full budget, so only finalizing remains. Re-invoking
        # after a crash or a walltime kill therefore never repeats work.
        print(f"log already holds {have} >= {budget} evaluations; finalizing only", flush=True)
        _open_partial(s, name, a.seed, "append")
        X, Y = None, None
    else:
        if resumable:
            print(f"resuming: {have} evaluations logged, checkpoint present", flush=True)
            CKPT["path"] = ckpt
            if a.stop_tol > 0:
                d = np.atleast_2d(np.loadtxt(part, delimiter=",", skiprows=1))
                TRACE["Y"].append(d[:, s["D"]:s["D"] + s["m"]])
                print(f"  convergence test seeded with {len(d)} logged evaluations", flush=True)
            _open_partial(s, name, a.seed, "append")
        else:
            if have:
                print(f"discarding {have} orphaned evaluations (no checkpoint to resume from)",
                      flush=True)
            if not a.no_checkpoint and a.method in CKPT_METHODS:
                CKPT["path"] = ckpt
                ckpt.unlink(missing_ok=True)
            _open_partial(s, name, a.seed, "fresh")
        X, Y = METHODS[a.method](s, budget, a.seed, a.max_workers)
    out = save(s, name, a.seed, X, Y, budget, a.force)
    PARTIAL["fh"].close()
    if out["n_eval"] >= budget:
        PARTIAL["path"].unlink(missing_ok=True)
        if CKPT["path"] is not None:
            CKPT["path"].unlink(missing_ok=True)
    else:
        # The optimizer returned normally below its allowance, i.e. its own termination criteria
        # fired (the budget is an allowance, not a mandate). Resuming cannot add evaluations, so the
        # run is final: record why and retire the transient state instead of leaving the run to be
        # resubmitted forever. A convergence stop is a deliberate outcome rather than a shortfall,
        # so it is recorded as such -- resume.sh reads this file and must not restart it.
        out["self_terminated"] = True
        rec = {"n_eval": out["n_eval"], "budget": budget, "shortfall": out["budget_shortfall"]}
        if STOP["reason"]:
            out["converged"] = STOP["reason"]
            rec.update(converged=STOP["reason"], hv_curve=STOP["curve"])
        (DATA / s["tag"] / f"{name}_seed{a.seed}.short").write_text(json.dumps(rec) + "\n")
        PARTIAL["path"].unlink(missing_ok=True)
        if CKPT["path"] is not None:
            CKPT["path"].unlink(missing_ok=True)
    if a.label:
        # Provenance sidecar for reference-building runs. The history CSV cannot say what population
        # produced it or why the search stopped, and both belong in the released record.
        (DATA / s["tag"] / f"{name}_seed{a.seed}.meta.json").write_text(json.dumps({
            "problem": a.problem, "method": a.method, "label": a.label, "seed": a.seed,
            "pop": a.pop or None, "budget_ceiling": budget, "gate": a.gate,
            "stop_tol": a.stop_tol, "stop_patience": a.stop_patience, "stop_min_gen": a.stop_min_gen, "stop_floor": STOP["floor"],
            "n_eval": out["n_eval"], "n_generations": len(STOP["curve"]) or None,
            "converged": STOP["reason"], "hv_curve": STOP["curve"] or None}, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
