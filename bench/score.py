"""Recompute the released attainment fractions from the released files.

Depends on numpy and nothing else. Neither XFOIL nor the AirDbM API is needed: this module reads the
published histories and summaries, so a data analyst can reproduce and extend every reported score.

The summaries carry two different hypervolumes, and mixing them up is the easiest mistake to make
with this benchmark:

``running_hv`` in a history CSV
    Dominated hypervolume of the rows so far in RAW objective units, referenced to the origin. It is a
    convergence trace, stored so a curve needs no recomputation, and it is the quantity ``hv_ref_abs``
    in ``mo_summary.json`` is comparable against.

``hv_ref_norm`` and ``frac_of_reference_hv``
    Dominated hypervolume after each objective is divided by the reference front's own maximum, with
    the reference point at ``REF_OFFSET`` below the normalized objective floor. This is the scored
    quantity, and it is what ``norm_hv`` below computes.

Range normalization is not cosmetic. Peak efficiency runs to a few hundred while stall margin runs to
a few tens of degrees, so an origin-referenced hypervolume on raw values would fix an arbitrary
exchange rate between one unit of efficiency and one degree of margin, and the ranking would follow
that choice. Dividing by the front maximum makes both axes dimensionless and comparable.

Run ``python bench/score.py`` to reproduce every ``frac_of_reference`` and ``frac_of_reference_hv`` in
the released summaries from the released histories, or import ``norm_hv``, ``score_so_run`` and
``score_mo_run`` to score your own optimizer on the same footing.
"""

import json
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
SO_METHODS = ("cmaes", "de", "ga", "pso", "sobol")
MO_METHODS = ("nsga2", "smsemoa", "moead", "optuna", "sobol")

# Offset of the hypervolume reference point below the normalized objective floor, in units of the
# per-problem normalizer. This is a fixed, versioned scoring choice; hypervolume values and method
# orderings can depend on it even when every method is scored against the same point.
REF_OFFSET = 0.05


def hv2d(Y, ref=(0.0, 0.0)):
    """Exact dominated hypervolume of a bi-objective MAXIMIZATION set above ``ref``.

    Sweeps the non-dominated staircase in order of decreasing first objective, which is exact in two
    dimensions and needs no external indicator package.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))[:, :2]
    r0, r1 = float(ref[0]), float(ref[1])
    Y = Y[(Y[:, 0] > r0) & (Y[:, 1] > r1)]
    if not len(Y):
        return 0.0
    Y = Y[np.lexsort((-Y[:, 1], -Y[:, 0]))]
    hv, best1 = 0.0, r1
    for y0, y1 in Y:
        if y1 > best1:
            hv += (y0 - r0) * (y1 - best1)
            best1 = y1
    return float(hv)


def non_dominated(Y):
    """The non-dominated subset of a bi-objective maximization set, duplicates collapsed.

    Sorting by the first objective descending and sweeping the running best of the second is exact in
    two dimensions and runs in n log n, which matters here: a reference front is pooled over more than
    a million evaluations.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))[:, :2]
    if not len(Y):
        return Y
    Y = Y[np.lexsort((-Y[:, 1], -Y[:, 0]))]
    keep = Y[:, 1] > np.maximum.accumulate(np.concatenate(([-np.inf], Y[:-1, 1])))
    return Y[keep]


def normalizer(front):
    """Per-objective divisor for a reference front: its own column maxima.

    A non-positive maximum would invert the axis, so it falls back to 1.0. That cannot occur on the
    released fronts, whose objectives are both non-negative and non-degenerate.
    """
    z = np.atleast_2d(np.asarray(front, dtype=float))[:, :2].max(axis=0)
    return np.where(z <= 0, 1.0, z)


def norm_hv(Y, z, ref_offset=REF_OFFSET):
    """Range-normalized dominated hypervolume of a bi-objective maximization set.

    ``Y`` is scored as a maximization set, ``z`` is the divisor from ``normalizer``, and the reference
    point sits at ``ref_offset`` in each normalized objective. This is the quantity behind
    ``hv_ref_norm`` and every ``frac_of_reference_hv``.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))[:, :2] / z
    # The reference point sits ref_offset BELOW the origin of the normalized objectives, i.e. 5% below
    # the objective floor. The origin is not the Pareto-front nadir. Every dominated box is widened by
    # the same constant margin on both axes.
    return hv2d(Y, ref=(-ref_offset, -ref_offset))


def score_so_run(y, y_ref):
    """Attainment fraction of one single-objective run: its best value over the released reference.

    The fraction reported per method in ``so_summary.json`` is the MEAN of this over the five seeds.
    """
    return float(np.max(np.asarray(y, dtype=float))) / float(y_ref)


def score_mo_run(Y, front, hv_ref_norm=None, ref_offset=REF_OFFSET):
    """Attainment fraction of a bi-objective run against a released reference front.

    ``Y`` is the run's full ``(n, 2)`` objective history and ``front`` the reference front from
    ``mo_summary.json``. Both are normalized by the front's maxima, so the score is the run's share of
    the reference hypervolume on a common, dimensionless footing.
    """
    z = normalizer(front)
    if hv_ref_norm is None:
        hv_ref_norm = norm_hv(front, z, ref_offset)
    return norm_hv(Y, z, ref_offset) / float(hv_ref_norm)


def tag(D, mach, reynolds, m):
    return f"{'SO' if m == 1 else 'MO'}_D{D}_Re{reynolds:.0e}_Ma{mach}"


def histories(d, method):
    """Released histories of one method in a problem directory, in seed order."""
    return sorted(Path(d).glob(f"{method}_seed*.csv.gz"))


def refstage_files(d):
    """Reference-stage evaluation files in a problem directory, in a stable order.

    These are the epsilon-constraint sweeps that improved each reference front beyond what the
    compared optimizers reached: ``_refine*`` and ``_frontfill*`` hold earlier batches of levels, and
    ``_trace*`` the systematic sweep. They are released alongside the histories so a front can be
    rederived, but they are pooled ONLY into the reference and never into a per-method statistic. The
    refiner is not one of the compared optimizers; it is part of how the target is built.
    """
    d = Path(d)

    def g(pat):
        return sorted(list(d.glob(pat)) + list(d.glob(pat + ".gz")),
                      key=lambda p: p.name.removesuffix(".gz"))

    return g("_refine*.csv") + g("_frontfill*.csv") + g("_epsgeq*.csv") + g("_trace*.csv")


def excluded_values():
    """Objective values the stability screen removed, keyed by (mach, reynolds).

    A value that no neighbor of its design reproduces is isolated: search cannot reach it and it
    cannot serve as a reference, so it is scored at the failure floor. The raw histories are left
    untouched and this list ships beside them, so either view can be reconstructed. It holds a single
    point in v0.3.0.
    """
    f = DATA / "robustness.json"
    if not f.is_file():
        return {}
    out = {}
    for e in json.load(open(f)).get("excluded", []):
        out.setdefault((round(e["mach"], 6), round(e["reynolds"], 3)), set()).add(
            tuple(round(float(v), 10) for v in e["y_released"]))
    return out


def _drop_excluded(Y, mach, reynolds, ex=None):
    ex = (excluded_values() if ex is None else ex).get(
        (round(mach, 6), round(reynolds, 3)), set())
    if not ex:
        return Y
    keep = [i for i, y in enumerate(Y) if tuple(np.round(y[:2], 10)) not in ex]
    return Y[keep]


def reference_front(pid, per=None, ex=None):
    """Rederive a released reference front from the released files.

    Chains up the nested dimensions exactly as the release builds them: the front at dimension k is
    the non-dominated set of this problem's histories, its reference-stage evaluations, and the front
    already established at dimension k-1 for the same condition. Chaining fronts rather than pooling
    all lower-dimension histories is what makes the reference monotone in D by construction, since a
    lower-dimensional design zero-pads to a valid higher-dimensional one and re-evaluates identically.
    """
    if per is None:
        per = json.load(open(DATA / "mo_summary.json"))["per_problem"]
    if ex is None:
        ex = excluded_values()
    c, n = pid.split("-")[2], int(pid.split("-")[3])
    front = None
    for k in range(1, n + 1):
        e = per[f"ADO-M-{c}-{k}"]
        d = DATA / tag(e["D"], e["mach"], e["reynolds"], 2)
        blocks = [] if front is None else [front]
        for f in [g for meth in MO_METHODS for g in histories(d, meth)] + refstage_files(d):
            a = np.atleast_2d(np.loadtxt(f, delimiter=",", skiprows=1))
            if a.size:
                blocks.append(_drop_excluded(a[:, e["D"]:e["D"] + 2], e["mach"], e["reynolds"], ex))
        front = non_dominated(np.vstack(blocks))
    return front


def verify_fronts(tol=1e-9):
    """Rederive every released reference front from released files and compare to the summary."""
    per = json.load(open(DATA / "mo_summary.json"))["per_problem"]
    ok = True
    for pid in per:
        got = reference_front(pid, per)
        want = np.asarray(per[pid]["front_Y"], dtype=float)
        a = set(map(tuple, np.round(got, 9)))
        b = set(map(tuple, np.round(want, 9)))
        same = a == b
        ok &= same
        print(f"{'  OK' if same else 'FAIL'}  {pid:12s} rederived={len(got):4d} "
              f"released={len(want):4d} missing={len(b - a):3d} extra={len(a - b):3d}")
    return ok


def verify(tol=1e-9):
    """Recompute every released attainment fraction and report the largest disagreement."""
    per = {1: json.load(open(DATA / "so_summary.json"))["per_problem"],
           2: json.load(open(DATA / "mo_summary.json"))["per_problem"]}
    ex = excluded_values()
    worst, n, bad = 0.0, 0, 0
    for m in (1, 2):
        for pid, e in per[m].items():
            D = e["D"]
            front = None if m == 1 else np.asarray(e["front_Y"], dtype=float)
            d = DATA / tag(D, e["mach"], e["reynolds"], m)
            for meth in (SO_METHODS if m == 1 else MO_METHODS):
                got = []
                for f in histories(d, meth):
                    a = np.atleast_2d(np.loadtxt(f, delimiter=",", skiprows=1))
                    Y = a[:, D:D + m]
                    Y = _drop_excluded(Y, e["mach"], e["reynolds"], ex)
                    got.append(score_so_run(Y[:, 0], e["y_ref"]) if m == 1
                               else score_mo_run(Y, front, e["hv_ref_norm"]))
                if not got:
                    print(f"MISS  {pid:12s} {meth}")
                    continue
                key = "frac_of_reference" if m == 1 else "frac_of_reference_hv"
                want = float(next(r[key] for r in e["stats"] if r["method"] == meth))
                # Both reported fractions are MEANS over the five seeds, not best-of-seed.
                mine = float(np.mean(got))
                dev = abs(mine - want)
                worst, n = max(worst, dev), n + 1
                bad += dev > tol
                print(f"{'  OK' if dev <= tol else 'FAIL'}  {pid:12s} {meth:8s} "
                      f"released={want:.6f} recomputed={mine:.6f} d={dev:.2e}")
    print(f"\n{n} method-problem pairs checked, {bad} outside {tol:g}, "
          f"largest disagreement {worst:.3e}")
    return bad == 0


if __name__ == "__main__":
    print("== reference fronts rederived from released files ==")
    a = verify_fronts()
    print("\n== attainment fractions recomputed from released files ==")
    b = verify()
    sys.exit(0 if (a and b) else 1)
