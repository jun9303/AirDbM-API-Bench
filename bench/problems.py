import json
import sys
from pathlib import Path

import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airdbm_core import TestAirfoils

DATA = Path(__file__).resolve().parent / "data"
CONDITIONS = {"2": (0.20, 1e6), "4": (0.40, 1e7)}
DIMS = {"1": 4, "2": 8, "3": 12}
CEILING = 350.0
SO_BUDGET = {4: 4096, 8: 8192, 12: 12288}
MO_BUDGET = {4: 4096, 8: 8192, 12: 12288}
SEEDS = (0, 1, 2, 3, 4)
SMOKE_SEED = 99
HV_REF = np.array([0.0, 0.0])
MIN_W = 1e-6

# STABILITY GATE -- a frozen part of the problem definition, not a property of a run.
#
# An objective value that no neighbor of the design reproduces is unattainable by search and must not
# stand (see verify.py). Screening every evaluation costs 4 extra evaluations each; screening whatever
# a run happens to hold in its archive would make the objective depend on the run. The gate instead
# fires on a CONSTANT -- a candidate the published reference solution set does not dominate -- so the
# same design is screened identically in every run, in any order, by any method.
#
# The gate fires when the published reference solution set does not dominate a candidate: for m=1 when
# the value exceeds the published best, for m=2 when no published front point dominates (y1, y2). The set
# is frozen in gate.json, so the same design is screened identically in every run, in any order, by any
# method. Measured on the released campaign it fires on 0.23% of bi-objective evaluations, about 57
# core-hours of probes, and covers every design the end-of-study screen removed.
#
# A frozen set cannot gate the campaign that produced it. The released runs were screened after the fact
# by verify.py, and gate.json freezes that outcome so every later study is gated against one published
# constant. The gate is either on, the default, or off.
GATE_FILE = Path(__file__).resolve().parent / "gate.json"
_GATE_CACHE: dict = {}


def gate_set(pid):
    """The frozen reference set for pid: ("best", float) for m=1, ("front", (n,2) array) for m=2."""
    if not _GATE_CACHE:
        with open(GATE_FILE) as fh:
            _GATE_CACHE.update(json.load(fh))
    if pid.split("-")[1] == "S":
        return "best", float(_GATE_CACHE["so_best"][pid])
    return "front", np.asarray(_GATE_CACHE["mo_front"][pid], dtype=float)


def gate_hits(Y, s, on=True):
    """Rows of Y that the gate arms the stability probe on."""
    Y = np.asarray(Y, dtype=float)
    Y = Y.reshape(len(Y), -1) if Y.ndim > 1 else Y.reshape(-1, 1)
    if not on:
        return np.zeros(len(Y), dtype=bool)
    kind, ref = gate_set(s["pid"])
    if kind == "best":
        return Y[:, 0] > ref
    # armed unless some published front point dominates the candidate
    dom = np.zeros(len(Y), dtype=bool)
    for p in ref:
        dom |= ((p[0] >= Y[:, 0]) & (p[1] >= Y[:, 1]) & ((p[0] > Y[:, 0]) | (p[1] > Y[:, 1])))
    return ~dom & (Y[:, 0] > 0)


SO_METHODS = ("cmaes", "de", "ga", "pso", "sobol")
MO_METHODS = ("nsga2", "smsemoa", "moead", "optuna", "sobol")


def problem_ids(n_obj=None):
    out = []
    for o in ("S", "M"):
        if n_obj is not None and o != ("S" if n_obj == 1 else "M"):
            continue
        for c in CONDITIONS:
            for d in DIMS:
                out.append(f"ADO-{o}-{c}-{d}")
    return out



FINAL_RE = re.compile(r"^_?[a-z0-9]+_seed\d+\.csv(\.gz)?$")


def history_files(d):
    """The finished evaluation histories in a problem directory, in a stable order.

    Every consumer of the released data must agree on what counts as a history, so the rule lives
    here rather than in each script. Two things are excluded deliberately:

      * `<run>.partial.csv`, the append-only log run.py writes while a run is in flight. It carries a
        different column layout (no derived best-so-far or hypervolume column), so a loose glob does
        not merely add unfinished data -- it misparses the columns it does read.
      * a superseded copy of a run. A re-run writes `<run>.csv` while the released version is still
        `<run>.csv.gz`; loading both would count one run twice and bias the pooled reference toward
        whichever copy happens to be better. That is an error, not something to resolve silently.
    """
    out, seen = [], {}
    for f in sorted(Path(d).glob("*_seed*.csv*")):
        if not FINAL_RE.match(f.name):
            continue
        stem = f.name[:-3] if f.name.endswith(".gz") else f.name
        if stem in seen:
            raise SystemExit(f"{d}: both {seen[stem].name} and {f.name} are present for the same "
                             f"run. Delete the superseded copy (the release convention is .csv.gz).")
        seen[stem] = f
        out.append(f)
    return out


def spec(pid):
    o, c, d = pid.split("-")[1:]
    mach, re_c = CONDITIONS[c]
    D = DIMS[d]
    m = 1 if o == "S" else 2
    return {"pid": pid, "m": m, "D": D, "mach": mach, "reynolds": re_c,
            "budget": (SO_BUDGET if m == 1 else MO_BUDGET)[D],
            # Objective form, dimension, then the flight condition -- the same shape for both forms,
            # so the two halves of the suite sort and read alike.
            "tag": f"{'SO' if m == 1 else 'MO'}_D{D}_Re{re_c:.0e}_Ma{mach:g}"}


def args_for(s, max_workers=None):
    a = {"airfoil_db_dir": str(ROOT / "airfoilDB"),
         "dbm_weight_range": [0.0, 1.0], "dbm_normalization": "ABS_SUM",
         "reynolds": float(s["reynolds"]), "mach": float(s["mach"]),
         "clcd_ceiling": CEILING, "xfoil_evaluation": True,
         "xfoil_backend": "apptainer", "xfoil_strict": False,
         "xfoil_retry": 1, "parallel": True}
    if max_workers is not None:
        a["max_workers"] = int(max_workers)
    return a


def screen(X, Y, s, max_workers=None, eps=1e-6, n_dir=4, rel_tol=0.02, on=True,
           verbose=True):
    """Floor any evaluation the gate arms that then fails the perturbation check.

    The verdict depends only on the design vector and the frozen reference set, so it is identical in
    every run. Returns the corrected objectives and the row indices that were floored.
    """
    from airdbm_core import VerifyDesigns
    X = np.atleast_2d(np.asarray(X, dtype=float))
    # reshape to (N, m) explicitly: atleast_2d would orient a single-objective column as one row
    Y = np.asarray(Y, dtype=float).reshape(len(X), -1).copy()
    idx = np.flatnonzero(gate_hits(Y, s, on=on))
    floored = []
    if len(idx):
        v = VerifyDesigns(X[idx], args=args_for(s, max_workers), m=s["m"], eps=eps, n_dir=n_dir,
                          rel_tol=rel_tol, objective=0)
        for i, r in zip(idx, v):
            if not r["robust"]:
                Y[i, :] = 0.0
                floored.append(int(i))
                if verbose:
                    print(f"    gate({version}): floored an unstable candidate "
                          f"({r['objectives']}), median neighbor {r['median_neighbor']}", flush=True)
    return (Y if s["m"] == 2 else Y[:, 0]), floored


def evaluate(X, s, max_workers=None):
    X = np.clip(np.atleast_2d(np.asarray(X, dtype=float)), 0.0, 1.0)
    m = s["m"]
    Y = np.zeros((len(X), m))
    ok = np.abs(X).sum(axis=1) >= MIN_W
    if not ok.any():
        return Y if m == 2 else Y[:, 0]
    res = TestAirfoils(X[ok], args=args_for(s, max_workers), m=m)
    idx = np.flatnonzero(ok)
    for j, r in enumerate(res):
        o = r.objectives
        v = list(o) if isinstance(o, (list, tuple)) else [o]
        c = float(v[0]) if v[0] is not None else np.nan
        if np.isfinite(c) and c > 0:
            Y[idx[j], 0] = c
            if m == 2:
                t = float(v[1]) if len(v) > 1 and v[1] is not None else np.nan
                Y[idx[j], 1] = max(0.0, t) if np.isfinite(t) else 0.0
    return Y if m == 2 else Y[:, 0]


def running_best(y):
    return np.maximum.accumulate(np.asarray(y, dtype=float).ravel())


def hypervolume(Y, ref=HV_REF):
    from pymoo.indicators.hv import HV
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    return float(HV(ref_point=-np.asarray(ref, dtype=float))(-Y))


def running_hv(Y):
    """Running dominated hypervolume of a 2-objective maximization history, referenced to (0, 0).

    Accumulated incrementally: the dominated region is a staircase, and only an evaluation that is
    non-dominated by the current staircase can change it. The obvious implementation -- recomputing
    the hypervolume of every prefix -- is quadratic, which is invisible at a panel run's 12k
    evaluations and fatal at a reference run's 400k, where it turns finalizing a run into hours of
    arithmetic. Verified to reproduce the released histories' stored column to better than 1e-9.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    out = np.empty(len(Y))
    st = []                                   # staircase, f1 descending
    hv = 0.0
    for i, (a, b) in enumerate(Y):
        if a > 0 and b > 0 and not any(p >= a and q >= b for p, q in st):
            st = [(p, q) for p, q in st if not (a >= p and b >= q)]
            st.append((a, b))
            st.sort(key=lambda t: -t[0])
            hv = sum((p - (st[j + 1][0] if j + 1 < len(st) else 0.0)) * q
                     for j, (p, q) in enumerate(st))
        out[i] = hv
    return out


def nd_indices(Y):
    """Indices of the non-dominated subset of a maximization set, strictly.

    For two objectives this is an exact sort-and-sweep: order by the first objective descending
    (ties broken by the second descending), then keep a point only if its second objective exceeds
    the running maximum. That is O(n log n) rather than the O(n^2) pairwise scan, which matters
    because pooling a problem's released runs reaches 3e5 points. Duplicated and weakly dominated
    points are dropped, neither of which affects a front or its hypervolume. Any other objective
    count falls back to the pairwise scan.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if Y.shape[1] != 2:
        nd = np.ones(len(Y), dtype=bool)
        for i in range(len(Y)):
            if not nd[i]:
                continue
            nd[np.all(Y[i] >= Y, axis=1) & np.any(Y[i] > Y, axis=1)] = False
            nd[i] = True
        return np.flatnonzero(nd)
    order = np.lexsort((-Y[:, 1], -Y[:, 0]))
    f1 = Y[order, 1]
    prev = np.concatenate(([-np.inf], np.maximum.accumulate(f1)[:-1]))
    return order[f1 > prev]


def non_dominated(Y):
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    return Y[nd_indices(Y)]


def weights_from_x(x):
    x = np.asarray(x, dtype=float)
    s = np.abs(x).sum()
    return x / s if s > MIN_W else x


def budget_for(s, method):
    del method
    return s["budget"]
