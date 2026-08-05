"""
airdbm_service.py
A JSON/HTTP surface for the AirDbM API and the AirDbM-Bench database.

The Python entry point (TestAirfoils) stays exactly as it is; this module wraps it in a
FastAPI application so a client in any language can post design vectors and receive objectives
as JSON. Nothing in airdbm_core.py is modified -- this module only imports from it.

Run it (installed):      airdbm-serve --host 0.0.0.0 --port 8000
Run it (repo checkout):  python airdbm_service.py
                         uvicorn airdbm_service:app --reload
Interactive docs:        http://<host>:<port>/docs      (OpenAPI schema at /openapi.json)

Endpoints
    GET  /healthz                                 liveness probe
    GET  /v1/meta                                 versions, defaults, objective modes
    GET  /v1/baselines                            the ordered DbM baseline library
    POST /v1/evaluate                             morph + evaluate a batch of design vectors
    GET  /v1/benchmark/problems                   the 12 ADO benchmark problems
    GET  /v1/benchmark/problems/{problem_id}      one problem: parameters + reference solution
    POST /v1/benchmark/problems/{problem_id}/evaluate
                                                  evaluate designs at a problem's frozen condition

Install the optional server dependencies with:  pip install "airdbm[server]"
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np

try:  # installed as part of the airdbm package
    from .airdbm_core import (  # type: ignore[import-not-found]
        EXPECTED_BASELINES, MACH, N_CRIT, NUM_POINTS_INTERP, REYNOLDS, TestAirfoils, __version__,
    )
except ImportError:  # repo checkout / flat module on sys.path
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from airdbm_core import (
        EXPECTED_BASELINES, MACH, N_CRIT, NUM_POINTS_INTERP, REYNOLDS, TestAirfoils, __version__,
    )

try:
    from fastapi import Body, FastAPI, HTTPException, Path as ApiPath, Query
    from pydantic import BaseModel, ConfigDict, Field, field_validator
except ImportError as exc:  # pragma: no cover - clearer than a bare ImportError at import time
    raise ImportError(
        "The AirDbM HTTP server needs the optional server dependencies. "
        'Install them with:  pip install "airdbm[server]"'
    ) from exc

API_VERSION = "v1"
MAX_CANDIDATES = 512          # a single request stays a single batch; bulk work belongs in a sweep
MAX_DIMENSION = len(EXPECTED_BASELINES)


def clcd_ceiling_for(reynolds: float) -> float:
    """Hard ceiling on (Cl/Cd)_max: 350 at every Reynolds number, matching
    airdbm_bench_so.clcd_ceiling_for. Part of the objective definition."""
    del reynolds  # deliberately Reynolds-independent
    return 350.0


# =============================================================================
# ARGUMENT PRESETS
# =============================================================================
# A preset is a named, frozen bundle of evaluation settings, so a client can
# reproduce a benchmark condition exactly without restating Mach, Reynolds and the Cl/Cd guard --
# and without risking a typo that silently scores a design under the wrong condition.
# CONDITION_PRESETS are the two flight conditions of the AirDbM-Bench (ADO) suite. They are the
# single source of truth in this module and are deliberately duplicated from the benchmark
# definition rather than imported, so the server also works in an installed wheel that does not
# ship the benchmark directory.

CONDITION_PRESETS: dict[str, dict[str, Any]] = {
    "wind_tunnel": {
        "mach": 0.20,
        "reynolds": 1e6,
        "label": "wind-tunnel scale model",
        "description": "215 mm chord at 68.1 m/s, sea-level ISA. ADO condition code 2.",
        "physical": {"altitude_m": 0.0, "velocity_ms": 68.1, "chord_m": 0.2146,
                     "temperature_k": 288.2, "density_kgm3": 1.225},
    },
    "regional_turboprop": {
        "mach": 0.40,
        "reynolds": 1e7,
        "label": "regional turboprop cruise",
        "description": "1.93 m chord at 126.4 m/s at 20 000 ft ISA; Saab 340B / ATR 42 class. "
                       "ADO condition code 4.",
        "physical": {"altitude_m": 6096.0, "velocity_ms": 126.4, "chord_m": 1.929,
                     "temperature_k": 248.5, "density_kgm3": 0.653},
    },
}

# Which condition preset each ADO problem code corresponds to.
_CONDITION_BY_CODE = {"2": "wind_tunnel", "4": "regional_turboprop"}
_DIM_BY_CODE = {"1": 4, "2": 8, "3": 12}


def _preset_args(name: str) -> dict[str, Any]:
    """Resolve a preset name to concrete evaluation settings.

    Accepts a condition preset name (wind_tunnel, regional_turboprop) or an ADO problem id
    (ADO-M-2-2), case-insensitively. A problem id resolves to its condition's settings.
    """
    key = name.strip()
    lowered = key.lower().replace("-", "_")
    if lowered in CONDITION_PRESETS:
        p = CONDITION_PRESETS[lowered]
        return {"mach": p["mach"], "reynolds": p["reynolds"],
                "clcd_ceiling": clcd_ceiling_for(p["reynolds"])}
    parts = key.upper().split("-")
    if len(parts) == 4 and parts[0] == "ADO" and parts[2] in _CONDITION_BY_CODE:
        p = CONDITION_PRESETS[_CONDITION_BY_CODE[parts[2]]]
        return {"mach": p["mach"], "reynolds": p["reynolds"],
                "clcd_ceiling": clcd_ceiling_for(p["reynolds"])}
    raise HTTPException(
        status_code=422,
        detail=f"Unknown preset '{name}'. Available: {sorted(CONDITION_PRESETS)} "
               f"or an ADO problem id such as 'ADO-M-2-2'. See GET /{API_VERSION}/presets.",
    )


# =============================================================================
# REQUEST / RESPONSE SCHEMAS
# =============================================================================
# Pydantic validates and documents them, and FastAPI
# publishes them as an OpenAPI 3.1 schema.

class EvaluationArgs(BaseModel):
    """Optional evaluation settings. Every field defaults to the interface default, so a
    minimal request body is just {"x": [[...]]}."""

    model_config = ConfigDict(extra="forbid")

    reynolds: Annotated[float, Field(gt=0)] = REYNOLDS
    mach: Annotated[float, Field(ge=0, lt=1)] = MACH
    n_crit: Annotated[float, Field(gt=0)] = N_CRIT
    dbm_weight_range: tuple[float, float] = (0.0, 1.0)
    dbm_normalization: Literal["ABS_SUM", "SUM", "none"] = "ABS_SUM"
    clcd_ceiling: Annotated[float, Field(gt=0)] | None = None
    alfa_start: float = 0.0
    alfa_end: float = 45.0
    xfoil_evaluation: bool = True
    xfoil_backend: Literal["apptainer", "native", "auto"] = "apptainer"
    xfoil_iter: Annotated[int, Field(ge=1)] = 200
    xfoil_timeout: Annotated[float, Field(ge=0)] = 60.0
    xfoil_retry: Annotated[int, Field(ge=0)] = 1
    parallel: bool = True
    max_workers: Annotated[int, Field(ge=1)] | None = None
    return_geometry: bool = False
    return_polar: bool = False

    def to_api_args(self) -> dict[str, Any]:
        """Translate to the keyword dictionary TestAirfoils expects."""
        args: dict[str, Any] = {
            "reynolds": self.reynolds,
            "mach": self.mach,
            "n_crit": self.n_crit,
            "dbm_weight_range": list(self.dbm_weight_range),
            "dbm_normalization": None if self.dbm_normalization == "none" else self.dbm_normalization,
            "alfa_start": self.alfa_start,
            "alfa_end": self.alfa_end,
            "xfoil_evaluation": self.xfoil_evaluation,
            "xfoil_backend": self.xfoil_backend,
            "xfoil_iter": self.xfoil_iter,
            "xfoil_timeout": self.xfoil_timeout,
            "xfoil_retry": self.xfoil_retry,
            "xfoil_strict": False,   # a failed candidate returns the objective floor, never a 500
            "parallel": self.parallel,
        }
        if self.max_workers is not None:
            args["max_workers"] = self.max_workers
        if self.clcd_ceiling is not None:
            args["clcd_ceiling"] = self.clcd_ceiling
        return args


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: list[list[float]] = Field(
        ...,
        description="Design matrix, N x D. Each row is one candidate; every entry lies in [0, 1]. "
                    "A candidate of dimension D uses the first D baselines of /v1/baselines.",
        examples=[[[0.94, 0.00, 0.81, 0.00]]],
    )
    m: Literal[1, 2] = Field(2, description="1 -> (Cl/Cd)_max only; 2 -> [(Cl/Cd)_max, stall margin].")
    preset: str | None = Field(
        None,
        description="Named argument preset: a benchmark condition ('wind_tunnel', "
                    "'regional_turboprop') or an ADO problem id ('ADO-M-2-2'). Sets Mach, Reynolds "
                    "and the Cl/Cd guard to that condition's frozen values. Any field you set "
                    "explicitly in args still wins, so a preset is a base, not a lock.",
        examples=["regional_turboprop"],
    )
    args: EvaluationArgs = Field(default_factory=EvaluationArgs)

    def resolved_args(self) -> EvaluationArgs:
        """Apply the preset as a base, then re-apply whatever the caller set explicitly."""
        if self.preset is None:
            return self.args
        base = _preset_args(self.preset)
        explicit = self.args.model_dump(exclude_unset=True)
        return EvaluationArgs(**{**base, **explicit})

    @field_validator("x")
    @classmethod
    def _check_matrix(cls, x: list[list[float]]) -> list[list[float]]:
        if not x:
            raise ValueError("x must contain at least one candidate")
        if len(x) > MAX_CANDIDATES:
            raise ValueError(f"x contains {len(x)} candidates; the per-request limit is {MAX_CANDIDATES}")
        width = len(x[0])
        if width == 0:
            raise ValueError("each candidate must have at least one design variable")
        if width > MAX_DIMENSION:
            raise ValueError(f"design dimension {width} exceeds the {MAX_DIMENSION} available baselines")
        for i, row in enumerate(x):
            if len(row) != width:
                raise ValueError(f"candidate {i} has length {len(row)}; expected {width} (matrix must be rectangular)")
            for v in row:
                if not (0.0 <= float(v) <= 1.0):
                    raise ValueError(f"candidate {i} has an entry outside [0, 1]")
        return x


class Geometry(BaseModel):
    """The morphed contour as the engine stores it: one closed curve in Selig order.

    A split into separate upper and lower arrays would misdescribe the data -- the engine keeps a
    single ordinate vector over the shared collocation stations, from the trailing edge forward over
    the upper surface and back along the lower.
    """

    x: list[float] = Field(..., description="Chordwise stations, Selig order (cosine-clustered).")
    y: list[float] = Field(..., description="Ordinates at those stations, same order and length as x.")
    n_points: int
    order: Literal["selig"] = "selig"


class CandidateResult(BaseModel):
    index: int = Field(..., description="Row of the submitted design matrix.")
    objectives: list[float] = Field(..., description="Length 1 if m=1, length 2 if m=2, in maximization sense.")
    cl_cd_max: float | None = None
    alpha_at_cl_cd_max: float | None = None
    cl_max: float | None = None
    alpha_stall: float | None = None
    delta_alpha: float | None = None
    converged: bool = Field(..., description="False when the solver failed and the objective floor was returned.")
    error: str | None = None
    name: str | None = None
    geometry: Geometry | None = None
    polar: dict[str, list[float]] | None = None


class EvaluateResponse(BaseModel):
    api_version: str = API_VERSION
    airdbm_version: str = __version__
    m: int
    n_candidates: int
    n_converged: int
    wall_time_s: float
    preset: str | None = Field(None, description="The preset applied to this request, if any.")
    conditions: dict[str, float] = Field(
        ..., description="The evaluation conditions actually used, after preset resolution.")
    results: list[CandidateResult]


class ProblemSummary(BaseModel):
    problem_id: str
    n_objectives: int
    dimension: int
    mach: float
    reynolds: float
    budget: int | None = None
    seeds: int | None = None


class ProblemDetail(ProblemSummary):
    objectives: list[str] | None = None
    reference: dict[str, Any] | None = None
    history_dir: str | None = None
    history_schema: Any | None = None


# =============================================================================
# BENCHMARK MANIFEST ACCESS
# =============================================================================
# Read-only, and optional: the evaluation endpoints work without it, only the
# /v1/benchmark endpoints need it. Searched in order, first hit wins, so a repo
# checkout resolves the released manifest and an installed package can be
# pointed at one with AIRDBM_BENCH_MANIFEST.

_MANIFEST_CANDIDATES = [
    Path(__file__).resolve().parent / "bench" / "data" / "manifest.json",
    Path(__file__).resolve().parent / "manifest.json",
    Path(__file__).resolve().parent / "airdbm_bench_manifest.json",
]


def _load_manifest() -> dict[str, Any] | None:
    env = os.environ.get("AIRDBM_BENCH_MANIFEST")
    paths = ([Path(env)] if env else []) + _MANIFEST_CANDIDATES
    for p in paths:
        try:
            if p.is_file():
                return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _require_manifest() -> dict[str, Any]:
    man = _load_manifest()
    if man is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark manifest not available in this installation. Set AIRDBM_BENCH_MANIFEST "
                   "to the path of airdbm_bench_manifest.json to enable the /v1/benchmark endpoints.",
        )
    return man


def _problem_entry(man: dict[str, Any], problem_id: str) -> dict[str, Any]:
    problems = man.get("problems", {})
    if problem_id not in problems:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown problem '{problem_id}'. Known problems: {sorted(problems)}",
        )
    return problems[problem_id]


def _n_obj(e: dict[str, Any]) -> int:
    """Objective count. The manifest writes it as 'M'; accept the obvious aliases too."""
    for k in ("M", "m", "n_objectives"):
        if e.get(k) is not None:
            return int(e[k])
    return 1


def _dim(e: dict[str, Any]) -> int:
    for k in ("D", "dimension"):
        if e.get(k) is not None:
            return int(e[k])
    return 0


def _obj_names(e: dict[str, Any]) -> list[str] | None:
    """The manifest stores a single human-readable 'objective' string; normalize to a list."""
    if isinstance(e.get("objectives"), list):
        return [str(v) for v in e["objectives"]]
    if e.get("objective") is not None:
        return [str(e["objective"])]
    return None


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(
    title="AirDbM API",
    version=__version__,
    summary="Design-by-Morphing airfoil generation with physics-in-the-loop XFOIL evaluation.",
    description=(
        "Post design vectors, get aerodynamic objectives back as JSON. The same evaluator that "
        "defines the AirDbM-Bench (ADO) benchmark problems, exposed over HTTP.\n\n"
        "Every evaluation is a real XFOIL solve, so requests are *slow by nature* "
        "(order of a second per candidate, more for pathological geometries). Submit a whole "
        "population in one call rather than one candidate per request."
    ),
    contact={"name": "AirDbM", "url": "https://doi.org/10.1093/jcde/qwaf124"},
    license_info={"name": "BSD 3-Clause"},
)


def _to_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _pack(results: list[Any], m: int, want_geometry: bool, want_polar: bool) -> list[CandidateResult]:
    packed: list[CandidateResult] = []
    for i, r in enumerate(results):
        xr = r.xfoil_result or {}
        obj = r.objectives
        if obj is None:
            vals = [0.0] * m
        elif np.isscalar(obj):
            vals = [float(obj)]
        else:
            vals = [float(v) for v in obj]
        clcd = _to_float(xr.get("cl_cd_max"))
        geom = None
        if want_geometry:
            xa = np.asarray(getattr(r.airfoil, "x_raw", []), dtype=float).ravel()
            ya = np.asarray(getattr(r.airfoil, "y_raw", []), dtype=float).ravel()
            geom = Geometry(x=xa.tolist(), y=ya.tolist(), n_points=int(xa.size))
        polar = None
        if want_polar and xr.get("alpha"):
            polar = {k: [float(v) for v in xr.get(k, [])] for k in ("alpha", "cl", "cd")}
        packed.append(CandidateResult(
            index=i,
            objectives=vals,
            cl_cd_max=clcd,
            alpha_at_cl_cd_max=_to_float(xr.get("alpha_at_cl_cd_max")),
            cl_max=_to_float(xr.get("cl_max")),
            alpha_stall=_to_float(xr.get("alpha_stall")),
            delta_alpha=_to_float(xr.get("delta_alpha")),
            converged=bool(vals and vals[0] > 0.0 and not xr.get("error")),
            error=(str(xr["error"]) if xr.get("error") else None),
            name=getattr(r.airfoil, "name", None),
            geometry=geom,
            polar=polar,
        ))
    return packed


# NOTE: these handlers are deliberately def, not async def. XFOIL evaluation is blocking and
# CPU-bound; Starlette runs sync handlers in a worker thread, so one slow request cannot stall
# the event loop and starve the health probe.

@app.get("/healthz", tags=["meta"], summary="Liveness probe")
def healthz() -> dict[str, str]:
    return {"status": "ok", "airdbm_version": __version__}


@app.get(f"/{API_VERSION}/meta", tags=["meta"], summary="Versions, defaults, and capabilities")
def meta() -> dict[str, Any]:
    return {
        "api_version": API_VERSION,
        "airdbm_version": __version__,
        "max_candidates_per_request": MAX_CANDIDATES,
        "max_dimension": MAX_DIMENSION,
        "objective_modes": {
            "1": ["cl_cd_max"],
            "2": ["cl_cd_max", "delta_alpha"],
        },
        "defaults": {"reynolds": REYNOLDS, "mach": MACH, "n_crit": N_CRIT,
                     "interp_points": NUM_POINTS_INTERP},
        "benchmark_available": _load_manifest() is not None,
    }


@app.get(f"/{API_VERSION}/baselines", tags=["meta"], summary="The ordered DbM baseline library")
def baselines() -> dict[str, Any]:
    return {
        "n_baselines": MAX_DIMENSION,
        "baselines": [{"index": i + 1, "name": n} for i, n in enumerate(EXPECTED_BASELINES)],
        "note": "A candidate of dimension D uses baselines 1..D, so the D=4, 8 and 12 spaces are nested.",
    }


@app.post(f"/{API_VERSION}/evaluate", tags=["evaluate"], response_model=EvaluateResponse,
          summary="Morph and evaluate a batch of design vectors")
def evaluate(req: Annotated[EvaluateRequest, Body()]) -> EvaluateResponse:
    X = np.asarray(req.x, dtype=float)
    args = req.resolved_args()
    t0 = time.time()
    try:
        results = TestAirfoils(X, args=args.to_api_args(), m=req.m)
    except Exception as exc:  # a batch-level failure is a server-side problem, not a bad request
        raise HTTPException(status_code=500, detail=f"evaluation failed: {exc}") from exc
    packed = _pack(results, req.m, args.return_geometry, args.return_polar)
    return EvaluateResponse(
        m=req.m,
        n_candidates=len(packed),
        n_converged=sum(1 for p in packed if p.converged),
        wall_time_s=round(time.time() - t0, 3),
        preset=req.preset,
        conditions={"reynolds": args.reynolds, "mach": args.mach, "n_crit": args.n_crit,
                    "clcd_ceiling": args.clcd_ceiling
                    if args.clcd_ceiling is not None else clcd_ceiling_for(args.reynolds)},
        results=packed,
    )


@app.get(f"/{API_VERSION}/presets", tags=["meta"],
         summary="Named argument presets for the benchmark conditions")
def presets() -> dict[str, Any]:
    """The frozen evaluation settings behind each benchmark condition, plus the problem ids that
    map onto them. Post {"preset": "<name>"} to /v1/evaluate to reproduce a condition exactly."""
    man = _load_manifest()
    known_ids = sorted(man.get("problems", {})) if man else []

    def _ids_for_code(code: str | None) -> list[str]:
        """Problem ids belonging to a condition. Requires the full ADO-<O>-<C>-<N> shape, so an
        id from a different scheme is never mis-assigned by field position."""
        if code is None:
            return []
        out = []
        for pid in known_ids:
            parts = pid.upper().split("-")
            if (len(parts) == 4 and parts[0] == "ADO" and parts[1] in ("S", "M")
                    and parts[2] == code and parts[3] in _DIM_BY_CODE):
                out.append(pid)
        return out

    conditions = {}
    for name, p in CONDITION_PRESETS.items():
        code = next((c for c, n in _CONDITION_BY_CODE.items() if n == name), None)
        conditions[name] = {
            "label": p["label"],
            "description": p["description"],
            "ado_condition_code": code,
            "args": {"mach": p["mach"], "reynolds": p["reynolds"],
                     "clcd_ceiling": round(clcd_ceiling_for(p["reynolds"]), 1)},
            "physical": p["physical"],
            "problem_ids": _ids_for_code(code),
        }
    return {
        "api_version": API_VERSION,
        "conditions": conditions,
        "dimension_codes": {k: v for k, v in _DIM_BY_CODE.items()},
        "usage": {
            "by_condition": {"preset": "regional_turboprop", "x": [[0.9, 0.0, 0.8, 0.0]], "m": 2},
            "by_problem_id": {"preset": "ADO-M-4-1", "x": [[0.9, 0.0, 0.8, 0.0]], "m": 2},
            "override": {"preset": "regional_turboprop", "args": {"n_crit": 7},
                         "x": [[0.9, 0.0, 0.8, 0.0]]},
        },
    }


@app.get(f"/{API_VERSION}/benchmark/problems", tags=["benchmark"],
         response_model=list[ProblemSummary], summary="List the ADO benchmark problems")
def list_problems() -> list[ProblemSummary]:
    man = _require_manifest()
    out = []
    for pid, e in sorted(man.get("problems", {}).items()):
        out.append(ProblemSummary(
            problem_id=pid,
            n_objectives=_n_obj(e),
            dimension=_dim(e),
            mach=float(e.get("mach", 0.0) or 0.0),
            reynolds=float(e.get("reynolds", REYNOLDS) or REYNOLDS),
            budget=e.get("budget"),
            seeds=e.get("seeds"),
        ))
    return out


@app.get(f"/{API_VERSION}/benchmark/problems/{{problem_id}}", tags=["benchmark"],
         response_model=ProblemDetail, summary="One problem: parameters and reference solution")
def get_problem(problem_id: Annotated[str, ApiPath(examples=["ADO-M-2-2"])]) -> ProblemDetail:
    man = _require_manifest()
    e = _problem_entry(man, problem_id)
    return ProblemDetail(
        problem_id=problem_id,
        n_objectives=_n_obj(e),
        dimension=_dim(e),
        mach=float(e.get("mach", 0.0) or 0.0),
        reynolds=float(e.get("reynolds", REYNOLDS) or REYNOLDS),
        budget=e.get("budget"),
        seeds=e.get("seeds"),
        objectives=_obj_names(e),
        reference=e.get("reference"),
        history_dir=e.get("history_dir"),
        history_schema=e.get("history_schema"),
    )


@app.post(f"/{API_VERSION}/benchmark/problems/{{problem_id}}/evaluate", tags=["benchmark"],
          response_model=EvaluateResponse,
          summary="Evaluate designs at a benchmark problem's frozen condition")
def evaluate_on_problem(
    problem_id: Annotated[str, ApiPath(examples=["ADO-M-2-2"])],
    x: Annotated[list[list[float]], Body(embed=True, examples=[[[0.94, 0.0, 0.81, 0.0, 0.0, 0.0, 0.01, 0.22]]])],
    max_workers: Annotated[int | None, Query(ge=1)] = None,
) -> EvaluateResponse:
    """Evaluate against a problem's own Mach, Reynolds number and objective count, so a submitted
    design is scored under exactly the protocol its released reference solution was obtained under."""
    man = _require_manifest()
    e = _problem_entry(man, problem_id)
    D = _dim(e)
    m = _n_obj(e)
    for i, row in enumerate(x):
        if len(row) != D:
            raise HTTPException(
                status_code=422,
                detail=f"candidate {i} has dimension {len(row)}; problem {problem_id} requires D={D}",
            )
    re_c = float(e.get("reynolds", REYNOLDS) or REYNOLDS)
    kw: dict[str, Any] = {
        "reynolds": re_c,                      # from the manifest, which is authoritative
        "mach": float(e.get("mach", 0.0) or 0.0),
        "clcd_ceiling": clcd_ceiling_for(re_c),
    }
    if max_workers is not None:
        kw["max_workers"] = max_workers
    return evaluate(EvaluateRequest(x=x, m=m, preset=problem_id, args=EvaluationArgs(**kw)))


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point (airdbm-serve)."""
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(
        prog="airdbm-serve", description="Serve the AirDbM API over HTTP (JSON).")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    ap.add_argument("--workers", type=int, default=1,
                    help="uvicorn worker processes; each one runs its own XFOIL pool")
    ap.add_argument("--reload", action="store_true", help="auto-reload on source changes (development)")
    ap.add_argument("--log-level", default="info",
                    choices=["critical", "error", "warning", "info", "debug", "trace"])
    a = ap.parse_args(argv)

    # --reload and multi-worker mode need an import string rather than the app object. Derive it
    # from this module's own name so it is correct both in a repo checkout ("airdbm_service") and
    # inside the installed package ("airdbm.airdbm_service").
    if a.reload or a.workers > 1:
        target: Any = f"{__name__}:app" if __name__ != "__main__" else "airdbm_service:app"
    else:
        target = app
    uvicorn.run(target, host=a.host, port=a.port, workers=a.workers,
                reload=a.reload, log_level=a.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
