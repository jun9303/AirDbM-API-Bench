"""
airdbm_core.py
The AirDbM evaluation engine: Design-by-Morphing airfoil generation and XFOIL scoring.

Maps an N x D matrix of candidate design vectors in [0, 1] onto Design-by-Morphing weights,
superposes the baseline library into a morphed geometry, validates and repairs that geometry,
and scores each candidate by driving XFOIL through a staged angle-of-attack scan -- returning the
aerodynamic objectives (Cl/Cd)_max and the stall margin. Candidates are evaluated in parallel
inside a robustness envelope: per-candidate timeout with process-group termination, a bounded
retry, and a non-strict mode in which a failed candidate scores the objective floor rather than
aborting the batch.

The single public entry point is TestAirfoils. For a JSON/HTTP surface over the same engine,
see airdbm_service.
"""

import os
import pickle
import shutil
import hashlib
import tempfile
import subprocess
import signal
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

try:
    from shapely.geometry import Polygon, LineString, Point, MultiPoint
    from shapely.validation import make_valid
except ImportError:
    Polygon = None
    print("Warning: Shapely library not available. Geometry correction will be bypassed.")

__version__ = "0.3.1"

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
NUM_POINTS_INTERP: int = 161
THETA_HALF: np.ndarray = np.linspace(0.0, np.pi, NUM_POINTS_INTERP // 2 + 1)
X_INTERP_HALF_ASC: np.ndarray = 0.5 * (1.0 - np.cos(THETA_HALF))[1:]
X_INTERP_HALF_DESC: np.ndarray = (0.5 * (1.0 - np.cos(THETA_HALF)))[::-1]
X_INTERP: np.ndarray = np.concatenate((X_INTERP_HALF_DESC, X_INTERP_HALF_ASC))

DATA_FOLDER: str = 'airfoilDB'

# Updated Default Expected Baselines order
EXPECTED_BASELINES: list[str] = [
    "E195  (11.82%)", "FX 79-W-660A", "GOE 531 AIRFOIL", "EPPLER 864 STRUT AIRFOIL",
    "RONCZ R1145MS MAIN ELEMENT", "CHEN AIRFOIL", "Griffith 30% Suction Airfoil", "S9104", 
    "AH 93-W-480B", "AH 81-K-144 W-F KLAPPE", "EPPLER 664 (EXTENDED) AIRFOIL", "SARATOV AIRFOIL"
]

# Cache to avoid reloading the database on repeated function calls
_CACHED_AIRFOIL_DB_DICT = None

# Geometry correction parameters
MIN_INTERIOR_THICKNESS: float = 1e-3  # Chord-normalized interior thickness floor for geometry correction
MIN_TRAILING_EDGE_THICKNESS: float = 1e-4  # Keep upper/lower TE endpoints from crossing

# XFOIL configuration
XFOIL_APP: str = 'bin/xfoil-ubuntu22.sif'  # Default apptainer image path for portable XFOIL execution
REYNOLDS: float = 1e6  # Default chord-based Reynolds number Re_c for XFOIL simulations
MACH: float = 0.0 # Default Mach number for XFOIL simulations (incompressible flow)
N_CRIT: float = 9.0 # Default N_crit for transition prediction in XFOIL
CD_BRANCH_TOL: float = 0.15       # max Cd disagreement between coarse and refined at a shared alpha
CD_DROP_TOL: float = 0.10         # max benign Cd fall AFTER the drag bucket (calibrated: 4.6% worst)
CD_BUCKET_DROP_TOL: float = 0.35  # max benign Cd fall WHILE still descending into the bucket
COARSE_ALPHA_STEP: float = 0.5 # Step size for initial coarse alpha scan in XFOIL
REFINED_ALPHA_STEP: float = 0.1 # Step size for refined alpha scan around identified centers in XFOIL
REFINE_ALPHA_WINDOW: float = 1.0 # Window size around identified centers for refined scanning in XFOIL

# =============================================================================
# UTILITIES
# =============================================================================
def moving_average(y_values: np.ndarray, window_size: int = 5) -> np.ndarray:
    if window_size < 1: return y_values
    return np.convolve(y_values, np.ones(window_size)/window_size, mode='same')

def _smooth_surface_preserve_endpoints(y_values: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Smooth a 1D surface while preserving endpoint values (LE/TE anchors)."""
    if window_size <= 1 or y_values.size < 3:
        return y_values.copy()

    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    padded = np.pad(y_values, (pad, pad), mode='edge')
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, kernel, mode='valid')

    smoothed[0] = y_values[0]
    smoothed[-1] = y_values[-1]
    return smoothed

def _prepare_surface_for_interp(x_surface: np.ndarray, y_surface: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Sort a surface in ascending x and merge duplicate x values using y-averaging.
    Returns None when insufficient points remain for interpolation.
    """
    if len(x_surface) < 2:
        return None

    order = np.argsort(x_surface)
    x_sorted = x_surface[order]
    y_sorted = y_surface[order]

    x_unique, inv = np.unique(x_sorted, return_inverse=True)
    if x_unique.size < 2:
        return None

    y_acc = np.zeros_like(x_unique, dtype=float)
    counts = np.zeros_like(x_unique, dtype=float)
    for i, grp in enumerate(inv):
        y_acc[grp] += y_sorted[i]
        counts[grp] += 1.0
    y_unique = y_acc / counts

    return x_unique, y_unique

def _resample_surface(x_surface: np.ndarray, y_surface: np.ndarray, x_target: np.ndarray) -> np.ndarray | None:
    prepared = _prepare_surface_for_interp(x_surface, y_surface)
    if prepared is None:
        return None
    x_unique, y_unique = prepared

    try:
        interpolator = PchipInterpolator(x_unique, y_unique)
    except ValueError:
        return None

    return interpolator(x_target)

def _enforce_min_interior_thickness(
    y_upper: np.ndarray,
    y_lower: np.ndarray,
    min_thickness: float = MIN_INTERIOR_THICKNESS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enforce minimum thickness on interior points only (exclude LE/TE anchors).
    This prevents local surface crossing and near-zero-thickness pockets.
    """
    y_u = y_upper.copy()
    y_l = y_lower.copy()

    if y_u.size < 3 or y_l.size < 3:
        return y_u, y_l

    thickness = y_u - y_l
    for i in range(1, len(thickness) - 1):
        if thickness[i] < min_thickness:
            delta = 0.5 * (min_thickness - thickness[i])
            y_u[i] += delta
            y_l[i] -= delta

    return y_u, y_l

def _resolve_xfoil_command(
    backend: str,
    apptainer_image: str,
    run_dir: str,
) -> tuple[list[str], str]:
    """
    Resolve command for XFOIL execution.
    Resolution order for auto mode is fixed: apptainer image, then system PATH 'xfoil'.
    """
    backend = (backend or 'auto').lower()
    if backend not in {'auto', 'native', 'apptainer'}:
        raise ValueError("xfoil_backend must be one of: 'auto', 'native', 'apptainer'.")

    def _apptainer_cmd() -> tuple[list[str], str]:
        apptainer_exe = shutil.which('apptainer')
        if not apptainer_exe:
            raise RuntimeError("Apptainer executable not found in system PATH.")

        image_abs = os.path.abspath(apptainer_image)
        if not os.path.exists(image_abs):
            raise RuntimeError(f"Apptainer image not found: {image_abs}")

        bind_arg = f"{run_dir}:{run_dir}"
        return [
            apptainer_exe,
            'run',
            '--bind', bind_arg,
            image_abs,
        ], 'apptainer'

    def _native_cmd() -> tuple[list[str], str]:
        system_xfoil = shutil.which('xfoil')
        if not system_xfoil:
            raise RuntimeError("XFOIL executable not found in system PATH.")
            
        xvfb_run = shutil.which('xvfb-run')
        if xvfb_run:
            # Safely wrap native execution in an isolated dummy display
            return [
                xvfb_run, 
                '-a', 
                '-s', '-screen 0 640x480x8', 
                system_xfoil
            ], 'native-xvfb'
        else:
            # Ultimate fallback if running natively and xvfb is missing
            return [system_xfoil], 'native-raw'

    if backend == 'apptainer':
        return _apptainer_cmd()

    if backend == 'native':
        return _native_cmd()

    # backend == 'auto': apptainer first, then native executable.
    try:
        return _apptainer_cmd()
    except Exception:
        return _native_cmd()

def _run_xfoil_aseq(
cmd: list[str],
    run_dir: str,
    coord_file: str,
    reynolds: float,
    mach: float,
    n_crit: float,
    repanel_n: int,
    max_iter: int,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    timeout_sec: float,
    warm_start: bool = False,
    anchor_alpha: float | None = None,
    diag: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # One polar file PER ATTEMPT. XFOIL's PACC *appends* to an existing file, and the PPAR->PANE
    # fallback re-runs the same session; sharing one path meant a first attempt that died after
    # PACC had opened the file left its rows behind for the fallback's parse to pick up.
    polar_stem = hashlib.sha256(os.urandom(32)).hexdigest()[:12]
    attempt_counter = {"n": 0}

    def _next_polar_name() -> str:
        attempt_counter["n"] += 1
        return f"p_{polar_stem}_{attempt_counter['n']}.out"

    # Pass BARE RELATIVE filenames to XFOIL. Popen runs with cwd=run_dir, and XFOIL truncates
    # filenames at 64 characters -- an absolute path under a long TMPDIR would be silently cut,
    # making the result depend on the temp-directory path length.
    coord_name = os.path.basename(coord_file)

    def _build_xfoil_input(use_ppar: bool, polar_name: str) -> str:
        commands = [f"LOAD {coord_name}"]

        if use_ppar and int(repanel_n) > 0:
            # XFOIL's paneling menu needs a SECOND blank line to exit after a parameter change.
            # With only one, the session fell out of sync and every PPAR attempt died with SIGFPE
            # (exit 136) -- so repanel_n was a no-op and every stage was run twice via the PANE
            # fallback, doubling the cost of the entire benchmark.
            commands.extend(["PPAR", f"N {int(repanel_n)}", "", ""])
            
        commands.extend([
            "PANE",
            "OPER",
            f"VISC {reynolds:.0f}",
            f"MACH {mach:.6f}",
            "VPAR",
            f"N {n_crit:.6f}",
            "",
            f"ITER {max_iter}"
        ])

        if warm_start:
            # Use the known-converged anchor, fallback to 0.0 if not provided
            start_ang = anchor_alpha if anchor_alpha is not None else 0.0
            commands.append(f"ALFA {start_ang:.3f}")
            
            # March toward alpha_start using a larger step (0.5) to build the BL quickly
            gap = alpha_start - start_ang
            if abs(gap) > 0.5:
                step = 0.5 if gap > 0 else -0.5
                # March up to just before the refined start to ensure a smooth handoff
                commands.append(f"ASEQ {start_ang:.3f} {alpha_start - step:.3f} {step:.3f}")

        commands.extend([
            "PACC",
            f"{polar_name}",
            "",
            f"ASEQ {alpha_start:.6f} {alpha_end:.6f} {alpha_step:.6f}",
            "PACC",
            "",
            "QUIT",
            ""
        ])
        
        return "\n".join(commands)

    def _run_once(exec_cmd: list[str], xfoil_input: str) -> subprocess.CompletedProcess:
        proc = subprocess.Popen(
            exec_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=run_dir,
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(
                input=xfoil_input, 
                timeout=timeout_sec if timeout_sec > 0.0 else None
            )
            return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
            
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            proc.communicate()
            raise
            
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            proc.communicate()
            raise

    polar_name = _next_polar_name()
    polar_path = os.path.join(run_dir, polar_name)

    try:
        proc = _run_once(cmd, _build_xfoil_input(use_ppar=True, polar_name=polar_name))

        # A nonzero exit is NOT evidence that paneling failed. Marching alpha to 45 degrees makes
        # XFOIL abort with SIGFPE (exit 136) on geometries it cannot carry that far, after having
        # written the converged low-incidence rows. Treating that as a PPAR failure spent a second
        # XFOIL launch that crashed identically. So only fall back to PANE when NOTHING was
        # produced -- the case that actually indicates the session never got going.
        if not os.path.exists(polar_path):
            if diag is not None:
                diag['ppar_fallbacks'] = int(diag.get('ppar_fallbacks', 0)) + 1
            polar_name = _next_polar_name()
            polar_path = os.path.join(run_dir, polar_name)
            proc = _run_once(cmd, _build_xfoil_input(use_ppar=False, polar_name=polar_name))

        # Record an abnormal exit. The polar it produced is kept: the abort point is a
        # deterministic function of the geometry (verified reproducible under CPU contention), so
        # the objective stays a reproducible function of the design -- the sweep simply ends where
        # XFOIL can no longer converge. This is recorded rather than hidden because it bounds the
        # stall-margin objective: a scan that aborts before the first lift maximum yields
        # delta_alpha = 0.
        if proc.returncode != 0 and diag is not None:
            diag['scan_aborts'] = int(diag.get('scan_aborts', 0)) + 1

    except subprocess.TimeoutExpired:
        # A killed scan leaves a PARTIALLY written polar on disk. Merging it would silently mix
        # points from an incompletely converged alpha march with converged ones -- and because the
        # timeout is measured in wall-clock time, whether that truncation happens depends on how
        # loaded the machine is. That made the same design score differently between runs (a
        # near-separation point could land on a branch with a strongly under-predicted Cd, which is
        # below the continuity filter's threshold and so passed it). XFOIL itself is
        # deterministic; accepting a truncated polar is what made the pipeline nondeterministic.
        # A stage that did not run to completion therefore contributes NOTHING: the caller keeps
        # the coarse points if a refined pass times out, and an incomplete coarse scan yields an
        # empty polar, which the retry/floor logic already handles.
        if diag is not None:
            diag['scan_timeouts'] = int(diag.get('scan_timeouts', 0)) + 1
        e = np.array([])
        return e, e, e, e, e

    return _parse_xfoil_polar(polar_path)

def _write_airfoil_for_xfoil(airfoil: 'Airfoil', file_path: str) -> None:
    with open(file_path, 'w') as f:
        f.write(f"{airfoil.name}\n")
        for x, y in zip(X_INTERP, airfoil.colloc_vec):
            f.write(f"{x:.10f} {y:.10f}\n")

def _parse_xfoil_polar(polar_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    """Read an XFOIL PACC polar as (alpha, Cl, Cd, Top_Xtr, Bot_Xtr).

    Columns 6-7 are the chordwise transition locations on the upper and lower surfaces. They carry
    no weight in the objective, but they say which boundary layer solution XFOIL settled on: a value
    at (or very near) 1.0 means the surface stayed laminar to the trailing edge. Two geometrically
    near-identical sections can differ here, which is what makes an isolated optimum possible, so
    the benchmark records them. A polar written by an older build without these columns yields NaN
    rather than dropping the row.
    """
    alpha_vals, cl_vals, cd_vals, xtr_top, xtr_bot = [], [], [], [], []
    if not os.path.exists(polar_file):
        e = np.array([])
        return e, e, e, e, e

    with open(polar_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                alpha = float(parts[0])
                cl = float(parts[1])
                cd = float(parts[2])
            except ValueError:
                continue
            try:
                xt, xb = float(parts[5]), float(parts[6])
            except (ValueError, IndexError):
                xt, xb = float('nan'), float('nan')
            alpha_vals.append(alpha)
            cl_vals.append(cl)
            cd_vals.append(cd)
            xtr_top.append(xt)
            xtr_bot.append(xb)
    return (np.array(alpha_vals), np.array(cl_vals), np.array(cd_vals),
            np.array(xtr_top), np.array(xtr_bot))

LAMINAR_XTR = 0.95   # a transition location at or beyond this is "no transition before the TE"


def _compute_polar_metrics(
    alpha: np.ndarray,
    cl: np.ndarray,
    cd: np.ndarray,
    reynolds: float,
    clcd_ceiling: float = 350.0,
    xtr_top: np.ndarray | None = None,
    xtr_bot: np.ndarray | None = None,
) -> dict:
    # 1. Dynamic Physical Drag Floor
    cd_min_physical = 2.656 / np.sqrt(reynolds)
    valid = cd > cd_min_physical

    # 2. Branch-continuity filter, following the polar upward from alpha_start.
    #
    # Physical basis. For a fixed section at fixed Re, Cd(alpha) falls into the drag bucket and
    # then rises monotonically as incidence increases and the boundary layer thickens and
    # eventually separates. Drag therefore may decrease only while the sweep is still descending
    # INTO the bucket. Once Cd has begun to rise, a subsequent fall means the solver has jumped to
    # a different boundary-layer solution -- typically the transition point relocating or a
    # laminar separation bubble bursting -- and every point after the jump lies on that other
    # branch, not on the one continued from alpha_start.
    #
    # Measured example this rule exists for: Cd = 0.01185 at alpha = 7.5 deg collapsing to
    # 0.00612 at 8.0 deg with Cl essentially unchanged (1.630 -> 1.600). Halving the drag in half
    # a degree while lift does not rise is not aerodynamics; it reported (Cl/Cd)_max = 253 on a
    # design whose continued branch gives about 171.
    #
    # Threshold. Calibrated on 49 randomly drawn designs at Ma = 0.20, Re_c = 1e6: the largest
    # consecutive drop seen on a well-behaved polar was 4.6% (median -0.2%, i.e. Cd usually
    # rises), while observed branch jumps were 44.7% and 48.4%. CD_DROP_TOL = 0.10 sits about 2x
    # above the worst benign case and 4x below the artifacts, so it separates them with a wide
    # margin. The previous rule -- a flat 50% drop anywhere in the polar -- missed both artifacts
    # by 1.6 and 5.3 percentage points and also permitted a fall after the drag rise had begun.
    CD_RISE_EPS = 0.02   # a rise of >2% marks the bucket as passed (ignores numerical jitter)
    risen = False
    for i in range(1, len(cd)):
        if cd[i - 1] <= 0 or cd[i] <= 0:
            continue
        if cd[i] > cd[i - 1] * (1.0 + CD_RISE_EPS):
            risen = True              # past the drag bucket: drag is now increasing with alpha
        elif risen and cd[i] < cd[i - 1] * (1.0 - CD_DROP_TOL):
            valid[i:] = False         # branch jump: this point and all beyond are off-branch
            break
        elif not risen and cd[i] < cd[i - 1] * (1.0 - CD_BUCKET_DROP_TOL):
            # Still descending toward the bucket, but no physical bucket descends this steeply
            # between neighbouring incidences either.
            valid[i:] = False
            break

    if not np.any(valid):
        return {
            'alpha': alpha.tolist(),
            'cl': cl.tolist(),
            'cd': cd.tolist(),
            'cl_cd': cl.tolist(),
            'cl_cd_max': 0.0,
            'alpha_at_cl_cd_max': 0.0,
            'cl_max': 0.0,
            'alpha_at_cl_max': 0.0,
            'alpha_stall': 0.0,
            'delta_alpha': 0.0,
            'xtr_top': [] if xtr_top is None else np.asarray(xtr_top).tolist(),
            'xtr_bot': [] if xtr_bot is None else np.asarray(xtr_bot).tolist(),
            'error': f'All valid Cd values were below physical threshold or solver continuity broke.'
        }

    cl_cd = np.full_like(cl, np.nan, dtype=float)
    cl_cd[valid] = cl[valid] / cd[valid]

    # Hard ceiling on Cl/Cd, applied at every Reynolds number. Configurable via the
    # 'clcd_ceiling' arg, default 350.
    cl_cd = np.clip(cl_cd, a_min=None, a_max=float(clcd_ceiling))

    # Smooth peak-related signals on valid polar entries to reduce zig-zag noise in alpha marching.
    # Note: The returned objective still reflects the peak index selected from the smoothed traces,
    # while the raw polar arrays in the result remain unchanged for inspection.
    cl_smooth = cl.copy()
    cl_cd_smooth = cl_cd.copy()
    valid_indices = np.where(valid)[0]
    if len(valid_indices) >= 5:
        window_size = min(5, len(valid_indices))
        if window_size % 2 == 0:
            window_size -= 1
        poly_order = min(3, window_size - 1)

        if window_size >= 3 and poly_order >= 1:
            cl_smooth[valid_indices] = savgol_filter(cl[valid_indices], window_size, poly_order)
            cl_cd_smooth[valid_indices] = savgol_filter(cl_cd[valid_indices], window_size, poly_order)

    # Note: Peak selection is intentionally based on the smoothed lift-to-drag curve to avoid
    # reporting a narrow raw-sample spike from noisy marching behavior.
    #
    # Both peak searches must see ONLY points that passed the validity mask (drag floor and
    # continuity filter). cl_cd is NaN at invalid points, so nanargmax already skips them; but
    # cl_smooth keeps the RAW value wherever smoothing was not applied, so a plain argmax over it
    # could select a point the mask had rejected. That fed cl_max, alpha_at_cl_max and the stall
    # march -- and hence delta_alpha, the second multi-objective objective -- from data the
    # protocol had already discarded.
    cl_valid = np.where(valid, cl_smooth, np.nan)
    idx_best = int(np.nanargmax(cl_cd_smooth))
    alpha_best = float(alpha[idx_best])
    idx_cl_max = int(np.nanargmax(cl_valid))

    # Conservative stall definition: first local Cl maximum while marching alpha upward starting from the peak lift-to-drag ratio.
    idx_stall = None
    if cl_smooth.size >= 3:
        start_idx = max(1, idx_best)
        for i in range(start_idx, len(cl_smooth) - 1):
            # Only a valid triple can define a local lift maximum; a rejected neighbour would
            # otherwise let filtered-out data set the stall incidence.
            if not (valid[i - 1] and valid[i] and valid[i + 1]):
                continue
            if cl_smooth[i] > cl_smooth[i - 1] and cl_smooth[i] >= cl_smooth[i + 1]:
                idx_stall = i
                break
    if idx_stall is None:
        idx_stall = max(idx_best, idx_cl_max)

    alpha_stall = float(alpha[idx_stall])
    raw_delta_alpha = float(alpha_stall - alpha_best)
    # The stall margin is non-negative by construction: stall is searched at or
    # beyond the peak lift-to-drag angle (idx_stall >= idx_best) on a strictly
    # ascending alpha grid. Clamp defensively and snap sub-degree floating-point
    # noise to exactly 0.0 so the reported margin is never a small negative value.
    delta_alpha = raw_delta_alpha if raw_delta_alpha > 1e-9 else 0.0

    # Transition state, recorded but NOT used in the objective. XFOIL can settle on either a long
    # laminar run or a transitioned boundary layer for two sections that differ below plotting
    # resolution, and the objective jumps between the two; the transition location is the cheapest
    # signal of which branch a given evaluation is on, and it comes free with the polar. A section
    # whose surfaces both stay laminar to the trailing edge at alpha = 0 is on the branch that a
    # small perturbation destroys.
    out = {
        'alpha': alpha.tolist(),
        'cl': cl.tolist(),
        'cd': cd.tolist(),
        'cl_cd': cl_cd.tolist(),
        'cl_cd_max': float(cl_cd[idx_best]),
        'alpha_at_cl_cd_max': alpha_best,
        'cl_max': float(cl[idx_cl_max]),
        'alpha_at_cl_max': float(alpha[idx_cl_max]),
        'alpha_stall': alpha_stall,
        'delta_alpha': delta_alpha,
    }
    if xtr_top is not None and xtr_bot is not None and len(xtr_top) == len(alpha):
        xt = np.asarray(xtr_top, dtype=float)
        xb = np.asarray(xtr_bot, dtype=float)
        i0 = int(np.argmin(np.abs(alpha)))          # the alpha = 0 row of the same march
        out['xtr_top'] = xt.tolist()
        out['xtr_bot'] = xb.tolist()
        out['xtr_top_at_zero'] = float(xt[i0])
        out['xtr_bot_at_zero'] = float(xb[i0])
        out['xtr_top_at_peak'] = float(xt[idx_best])
        out['xtr_bot_at_peak'] = float(xb[idx_best])
        both = max(float(xt[i0]), float(xb[i0]))
        out['fully_laminar_at_zero'] = bool(np.isfinite(both)
                                            and min(float(xt[i0]), float(xb[i0])) >= LAMINAR_XTR)
    return out

def run_xfoil_evaluation(airfoil: 'Airfoil', xfoil_config: dict, m: int = 1) -> dict:
    """
    Execute XFOIL and compute polar metrics without truncating post-stall data.
    """
    apptainer_image = str(xfoil_config.get('apptainer_image', XFOIL_APP))
    backend = str(xfoil_config.get('xfoil_backend', 'auto'))

    reynolds = float(xfoil_config.get('reynolds', REYNOLDS))
    mach = float(xfoil_config.get('mach', MACH))
    n_crit = float(xfoil_config.get('n_crit', N_CRIT))
    
    repanel_n = int(xfoil_config.get('repanel_n', 160))
    max_iter = int(xfoil_config.get('xfoil_iter', 200))
    clcd_ceiling = float(xfoil_config.get('clcd_ceiling', 350.0))
    timeout_sec = float(xfoil_config.get('xfoil_timeout', 60.0))

    alpha_start = float(xfoil_config.get('alfa_start', 0.0))
    alpha_end = float(xfoil_config.get('alfa_end', 45.0))
    
    with tempfile.TemporaryDirectory(prefix='xfoil_run_') as run_dir:
        coord_file = os.path.join(run_dir, 'airfoil.dat')

        _write_airfoil_for_xfoil(airfoil, coord_file)
        cmd, runner = _resolve_xfoil_command(
            backend=backend,
            apptainer_image=apptainer_image,
            run_dir=run_dir,
        )

        # Counts alpha-scan stages killed by the wall-clock timeout. Such a stage contributes no
        # points (its partial polar is discarded, not merged), so a nonzero count means this
        # evaluation was scored from fewer stages than intended -- surfaced here so it is auditable
        # rather than silent.
        scan_diag: dict = {}

        # Stage-1 coarse scan: No warm start needed if starting from 0 (or close to it)
        alpha_c, cl_c, cd_c, xt_c, xb_c = _run_xfoil_aseq(
            cmd=cmd,
            run_dir=run_dir,
            coord_file=coord_file,
            reynolds=reynolds,
            mach=mach,
            n_crit=n_crit,
            repanel_n=repanel_n,
            max_iter=max_iter,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            alpha_step=COARSE_ALPHA_STEP,
            timeout_sec=timeout_sec,
            warm_start=False,
            diag=scan_diag
        )

        valid_c = cd_c > 0
        refine_centers = []

        # Attempt to find centers if the coarse scan yielded positive drag data
        if np.any(valid_c):
            cl_cd_c = np.full_like(cl_c, np.nan, dtype=float)
            cl_cd_c[valid_c] = cl_c[valid_c] / cd_c[valid_c]
            
            cl_c_smooth = cl_c.copy()
            cl_cd_c_smooth = cl_cd_c.copy()
            
            valid_indices = np.where(valid_c)[0]
            if len(valid_indices) >= 5: 
                window_size = 5 
                poly_order = 3  
                cl_c_smooth[valid_indices] = savgol_filter(cl_c[valid_indices], window_size, poly_order)
                cl_cd_c_smooth[valid_indices] = savgol_filter(cl_cd_c[valid_indices], window_size, poly_order)

            # Refine around BOTH the peak-efficiency and the maximum-lift incidence regardless of
            # how many objectives the caller asked for. Making this depend on m made
            # (Cl/Cd)_max itself depend on m: the maximum-lift window is only scanned when m=2,
            # and its 0.1-degree points overwrite the coarse 0.5-degree ones, so a coarse point
            # near stall could stand as the efficiency peak under m=1 and be corrected under m=2.
            # Measured on one design: at alpha=8.5 the coarse pass gave Cd=0.00669 -> Cl/Cd=237.9,
            # while the warm-started refined pass gave Cd=0.01465 -> Cl/Cd=112.6. The
            # single-objective and bi-objective suites would then not share one protocol, which
            # the benchmark's nesting argument requires.
            alpha_center_clcd = float(alpha_c[int(np.nanargmax(cl_cd_c_smooth))])
            alpha_center_clmax = float(alpha_c[int(np.argmax(cl_c_smooth))])
            refine_centers = sorted({alpha_center_clcd, alpha_center_clmax})

        # Initialize the merge dictionary with whatever coarse data we got (even if empty)
        merged = {
            round(float(a), 6): (float(a), float(c_l), float(c_d), float(xt), float(xb))
            for a, c_l, c_d, xt, xb in zip(alpha_c, cl_c, cd_c, xt_c, xb_c)
        }

        # Stage-2/3 refined scans (This loop will safely skip if refine_centers is empty).
        #
        # The reported peak must come from a REFINED pass. A coarse 0.5-degree scan jumps to a high
        # incidence without a converged boundary-layer history, and near stall it can report a drag
        # far below what a warm-started 0.1-degree march gives at the same angle. If such a point
        # falls outside every refined window it survives the merge and can become the reported
        # optimum -- an artefact of scan resolution, not of the design. So after refining, we check
        # where the merged peak actually lies and refine there too, repeating until the peak is
        # covered by a refined window (bounded, so a pathological polar cannot loop forever).
        refined_windows: list[tuple[float, float]] = []
        pending = list(refine_centers)
        MAX_EXTRA_REFINEMENTS = 3
        extra_used = 0

        def _covered(a: float) -> bool:
            return any(lo - 1e-9 <= a <= hi + 1e-9 for lo, hi in refined_windows)

        while pending:
            center = pending.pop(0)
            a0 = max(alpha_start, center - REFINE_ALPHA_WINDOW)
            a1 = min(alpha_end, center + REFINE_ALPHA_WINDOW)
            if a1 <= a0 or _covered(center):
                continue

            # The fine sweep MARCHES FROM THE SWEEP ORIGIN at the fine step, in one continuous
            # ASEQ, rather than warm-starting near the window and switching step size on the way.
            #
            # XFOIL offers no way to resume the coarse session's boundary-layer state, so a fine
            # pass is necessarily its own march. The previous approach re-marched at 0.5 deg from
            # an anchor and then switched to 0.1 deg at the window edge; mixing step sizes mid-march
            # changes the convergence path, which is exactly how a fine pass can end up on a
            # different branch from the coarse one. Sweeping from alpha_start at a single fine step
            # makes the fine solution the continuation of the SAME alpha = 0 evaluation, by
            # construction, and additionally gives many shared incidences for the consistency
            # check below rather than only those inside the window.
            #
            # Cost: the fine range reaches only to the window end (typically under 10 deg), not to
            # alpha_end (45 deg), so this is a modest increase over the previous fine stage -- far
            # cheaper than sweeping the whole polar at the fine step.
            alpha_r, cl_r, cd_r, xt_r, xb_r = _run_xfoil_aseq(
                cmd=cmd,
                run_dir=run_dir,
                coord_file=coord_file,
                reynolds=reynolds,
                mach=mach,
                n_crit=n_crit,
                repanel_n=repanel_n,
                max_iter=max_iter,
                alpha_start=alpha_start,
                alpha_end=a1,
                alpha_step=REFINED_ALPHA_STEP,
                timeout_sec=timeout_sec,
                warm_start=False,
                diag=scan_diag
            )

            # BRANCH-CONSISTENCY CHECK.
            #
            # The fine pass is its own XFOIL session and marches from alpha_start at the fine
            # step, so it should trace the same branch as the coarse sweep. It can still diverge --
            # a finer step can resolve a transition event the coarse grid stepped over -- and
            # because fine points OVERWRITE coarse ones at shared incidences, a divergent pass
            # would splice a second branch into the polar continued from alpha_start.
            #
            # At an incidence both passes evaluated, the same airfoil at the same Re and Mach must
            # give the same drag unless the two are on different branches. So compare them there:
            # if they differ beyond CD_BRANCH_TOL, the refined window is off-branch and is
            # DISCARDED, leaving the coarse points -- which are the branch continued from
            # alpha_start, and therefore authoritative -- in place.
            coarse_at = {round(float(a), 6): float(c) for a, c in zip(alpha_c, cd_c) if c > 0}
            mismatch = None
            for a, c_d in zip(alpha_r, cd_r):
                if c_d <= 0:
                    continue
                key = round(float(a), 6)
                c_ref = coarse_at.get(key)
                if c_ref is None or c_ref <= 0:
                    continue
                rel = abs(float(c_d) - c_ref) / c_ref
                if rel > CD_BRANCH_TOL:
                    mismatch = (float(a), c_ref, float(c_d), rel)
                    break

            if mismatch is not None:
                scan_diag['branch_mismatches'] = int(scan_diag.get('branch_mismatches', 0)) + 1
                # Do not merge, and do not mark the window refined: the coarse branch stands.
                continue

            for a, c_l, c_d, xt, xb in zip(alpha_r, cl_r, cd_r, xt_r, xb_r):
                merged[round(float(a), 6)] = (float(a), float(c_l), float(c_d),
                                              float(xt), float(xb))
            # Only a stage that actually returned points counts as refined. Marking a killed stage
            # as "covered" would disarm the peak-must-be-refined check below in exactly the case it
            # exists for: the coarse point would stand as the reported optimum, and the value would
            # then depend on machine load.
            if len(alpha_r) > 0:
                refined_windows.append((alpha_start, a1))

            # Where does the peak sit now? If it is still on an unrefined coarse point, refine it.
            if not pending and extra_used < MAX_EXTRA_REFINEMENTS and merged:
                rows = sorted(merged.values(), key=lambda r: r[0])
                aa = np.array([r[0] for r in rows]); cc = np.array([r[1] for r in rows])
                dd = np.array([r[2] for r in rows])
                ok = dd > 0
                if np.any(ok):
                    ratio = np.full_like(cc, np.nan, dtype=float)
                    ratio[ok] = cc[ok] / dd[ok]
                    for peak_alpha in ({float(aa[int(np.nanargmax(ratio))]),
                                        float(aa[int(np.argmax(cc))])}):
                        if not _covered(peak_alpha):
                            pending.append(peak_alpha)
                            extra_used += 1
        if extra_used:
            scan_diag['extra_refinements'] = extra_used

        # Convert the merged dictionary back to arrays
        merged_rows = sorted(merged.values(), key=lambda row: row[0])
        alpha = np.array([r[0] for r in merged_rows], dtype=float)
        cl = np.array([r[1] for r in merged_rows], dtype=float)
        cd = np.array([r[2] for r in merged_rows], dtype=float)
        xtr_top = np.array([r[3] for r in merged_rows], dtype=float)
        xtr_bot = np.array([r[4] for r in merged_rows], dtype=float)

        # Only fail if absolutely NO data was gathered across all 3 stages
        if len(alpha) == 0:
            metrics = {
                'alpha': [],
                'cl': [],
                'cd': [],
                'cl_cd': [],
                'cl_cd_max': 0.0,          # Worst possible efficiency
                'alpha_at_cl_cd_max': 0.0,
                'cl_max': 0.0,             # Worst possible lift
                'alpha_at_cl_max': 0.0,
                'alpha_stall': 0.0,
                'delta_alpha': 0.0,        # Worst possible stall margin
                'error': 'All XFOIL scans failed to converge.'
            }
        else:
            metrics = _compute_polar_metrics(alpha, cl, cd, reynolds,
                                             clcd_ceiling=clcd_ceiling,
                                             xtr_top=xtr_top, xtr_bot=xtr_bot)

        # Attach execution metadata regardless of success or failure
        metrics['runner'] = runner
        metrics['command'] = cmd
        metrics['scan_scheme'] = {
            'coarse_step': COARSE_ALPHA_STEP,
            'refined_step': REFINED_ALPHA_STEP,
            'refine_window': REFINE_ALPHA_WINDOW,
        }
        metrics['scan_timeouts'] = int(scan_diag.get('scan_timeouts', 0))
        metrics['extra_refinements'] = int(scan_diag.get('extra_refinements', 0))
        # Nonzero means the PPAR paneling command failed and the PANE fallback was used, so the
        # requested repanel_n was NOT applied and that stage cost two XFOIL launches.
        metrics['ppar_fallbacks'] = int(scan_diag.get('ppar_fallbacks', 0))
        # Refined windows discarded because they differed from the coarse branch at a shared
        # incidence (the refined session had settled on a different boundary-layer solution).
        metrics['branch_mismatches'] = int(scan_diag.get('branch_mismatches', 0))
        # Stages where XFOIL aborted (typically SIGFPE deep in the post-stall march). The rows it
        # produced are used; this counter makes the truncated alpha range auditable.
        metrics['scan_aborts'] = int(scan_diag.get('scan_aborts', 0))
        
        return metrics

# =============================================================================
# AIRFOIL CORE CLASSES & FILE I/O
# =============================================================================
class Airfoil:
    def __init__(self, airfoil_id: int, name: str, colloc_vec: np.ndarray, x_raw: np.ndarray, y_raw: np.ndarray):
        self.airfoil_id = airfoil_id
        self.name = name
        self.colloc_vec = colloc_vec
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.xfoil_result = None

    def __str__(self) -> str:
        return f"Airfoil(ID={self.airfoil_id}, Name='{self.name}', Raw Data Points={len(self.x_raw)})"

    def get_raw_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x_raw, self.y_raw

    def get_interpolated_data(self) -> np.ndarray:
        return self.colloc_vec

    def plot(self, save_path: str | None = None) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_raw, self.y_raw, 'o', label='Raw Data', alpha=0.7, markersize=4)
        # The contour runs from the upper trailing edge around the nose to the lower trailing edge,
        # so repeat the first point to close the section across the trailing-edge face.
        plt.plot(np.append(X_INTERP, X_INTERP[0]),
                 np.append(self.colloc_vec, self.colloc_vec[0]),
                 '-', label='Interpolated Data', linewidth=1.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Airfoil: {self.name}", fontsize=14)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        # plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def read_airfoil_file(file_path: str) -> tuple[str, np.ndarray, np.ndarray]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        name = lines[0].strip()
        coords_list = []
        is_goe451 = name.startswith("GOE 451")

        for line_content in lines[1:]:
            line_content = line_content.strip()
            if line_content.startswith("#") or not line_content:
                continue
            coords_list.append(list(map(float, line_content.split())))
        
        coords_np = np.array(coords_list)
        x_original, y_original = coords_np[:, 0], coords_np[:, 1]

        if is_goe451: # DB has error for this specific airfoil
            y_original[x_original == 0.0249500] = 0.0192620 

        x_min, x_max = x_original.min(), x_original.max()
        if x_max > x_min:
            x_original = (x_original - x_min) / (x_max - x_min)

    leading_edge_indices = np.where(np.isclose(x_original, 0, atol=1e-8))[0]

    if len(leading_edge_indices) == 0:
        for i in range(1, len(x_original)):
            if x_original[i-1] > 0 and x_original[i] < 0:
                y_le = np.interp(0, [x_original[i-1], x_original[i]], [y_original[i-1], y_original[i]])
                x_original = np.insert(x_original, i, 0)
                y_original = np.insert(y_original, i, y_le)
                break
    elif len(leading_edge_indices) == 2:
        upper_le_idx, lower_le_idx = leading_edge_indices
        y_avg_le = (y_original[upper_le_idx] + y_original[lower_le_idx]) / 2.0
        x_original[upper_le_idx] = 1e-6 
        x_original[lower_le_idx] = 1e-6
        insert_idx = min(upper_le_idx, lower_le_idx) + 1
        x_original = np.insert(x_original, insert_idx, 0)
        y_original = np.insert(y_original, insert_idx, y_avg_le)
        
    return name, x_original, y_original

def interp_airfoil(x_coords: np.ndarray, y_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(x_coords) != len(y_coords) or len(x_coords) < 4:
        return None

    min_x = np.min(x_coords)
    le_indices = np.where(np.isclose(x_coords, min_x, atol=1e-8))[0]
    if le_indices.size == 0:
        return None

    le_start, le_end = int(le_indices[0]), int(le_indices[-1])
    x_upper_raw, y_upper_raw = x_coords[:le_start + 1].copy(), y_coords[:le_start + 1].copy()
    x_lower_raw, y_lower_raw = x_coords[le_end:].copy(), y_coords[le_end:].copy()

    if len(x_lower_raw) < 2 or len(x_upper_raw) < 2:
        return None

    y_le = np.mean(y_coords[le_indices])
    x_upper_raw[-1], y_upper_raw[-1] = min_x, y_le
    x_lower_raw[0], y_lower_raw[0] = min_x, y_le

    prepared_upper = _prepare_surface_for_interp(x_upper_raw, y_upper_raw)
    prepared_lower = _prepare_surface_for_interp(x_lower_raw, y_lower_raw)
    if prepared_upper is None or prepared_lower is None:
        return None

    x_upper, y_upper = prepared_upper
    x_lower, y_lower = prepared_lower

    try:
        f_upper = PchipInterpolator(x_upper, y_upper)
        f_lower = PchipInterpolator(x_lower, y_lower)
    except ValueError:
        return None

    y_interp_upper = f_upper(X_INTERP[:NUM_POINTS_INTERP // 2 + 1])
    y_interp_lower = f_lower(X_INTERP[NUM_POINTS_INTERP // 2:])

    # Ensure a single consistent LE point where upper and lower branches meet.
    y_le_interp = 0.5 * (y_interp_upper[-1] + y_interp_lower[0])
    y_interp_upper[-1] = y_le_interp
    y_interp_lower[0] = y_le_interp

    y_interp_combined = np.concatenate((y_interp_upper, y_interp_lower[1:]))
    return X_INTERP, y_interp_combined

def _load_cached_db(pickle_file_path: str) -> list['Airfoil'] | None:
    """
    Load the cached database pickle, returning None when it is missing, empty,
    or corrupt so the caller can rebuild from the raw .dat files instead.

    Under multiprocessing, a pickle can be observed while another worker (or a
    packaged-data copy step) is still writing it, yielding a partially written
    file. Reading such a file raises 'pickle data was truncated' / EOFError.
    Treating any unreadable pickle as a cache miss keeps parallel runs safe.
    """
    if not os.path.exists(pickle_file_path):
        return None
    try:
        if os.path.getsize(pickle_file_path) == 0:
            return None
        with open(pickle_file_path, 'rb') as f:
            db = pickle.load(f)
    except Exception:
        # Treat ANY failure to read/deserialize as a cache miss so the caller
        # rebuilds from the raw .dat files rather than crashing. Besides the
        # common prefix-truncation errors (UnpicklingError/EOFError), a pickle
        # with corrupted interior bytes can raise MemoryError, TypeError,
        # OverflowError, IndexError, etc. from a bad length/opcode field.
        return None
    if not isinstance(db, list) or not db:
        return None
    return db


def _build_db_from_dat_files(data_folder: str) -> tuple[list['Airfoil'], bool]:
    """
    Rebuild the airfoil database directly from Selig-format .dat files.

    Returns (airfoils_db, complete). 'complete' is False when a .dat file could
    not be read (e.g. it is still being copied by a concurrent packaged-data
    populate step); the caller then avoids persisting a partial cache.
    """
    airfoils_db: list['Airfoil'] = []
    if not os.path.isdir(data_folder):
        return airfoils_db, True

    incompatible_files = {'30p-30n.dat', 'naca1.dat'}
    affile_list_filtered = [
        f for f in os.listdir(data_folder)
        if f not in incompatible_files and f.endswith('.dat')
    ]

    complete = True
    for idx, filename in enumerate(sorted(affile_list_filtered)):
        file_full_path = os.path.join(data_folder, filename)
        try:
            af_name, x_o, y_o = read_airfoil_file(file_full_path)
        except (OSError, ValueError, IndexError):
            # Unreadable or still being copied: mark the build incomplete so the
            # result is not cached, then let a later load rebuild the full set.
            complete = False
            continue

        interp_result = interp_airfoil(x_o, y_o)
        if interp_result is None:
            continue

        _, y_interp = interp_result
        airfoils_db.append(
            Airfoil(airfoil_id=idx, name=af_name, colloc_vec=y_interp, x_raw=x_o, y_raw=y_o)
        )

    return airfoils_db, complete


def _atomic_pickle_dump(obj: object, pickle_file_path: str) -> None:
    """
    Write a pickle atomically so concurrent readers never observe a partial file.

    The object is written to a uniquely named temp file in the destination
    directory, flushed and fsynced, then os.replace()'d into place (an atomic
    rename on POSIX). This eliminates the interleaved/truncated writes that occur
    when several parallel workers persist the cache at once. Failures here are
    non-fatal: the in-memory database is already available to the caller.
    """
    directory = os.path.dirname(pickle_file_path) or '.'
    try:
        fd, tmp_path = tempfile.mkstemp(prefix='_db.', suffix='.pkl.tmp', dir=directory)
    except OSError:
        return
    published = False
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, pickle_file_path)
        published = True
    except Exception:
        # Persisting the cache is best-effort; the in-memory DB is already
        # available to the caller, so never let a write failure (OSError,
        # MemoryError, PicklingError, ...) propagate.
        pass
    finally:
        if not published:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_airfoil_database_from_files(data_folder: str) -> list[Airfoil]:
    pickle_file_path = os.path.join(data_folder, '_db.pkl')

    cached = _load_cached_db(pickle_file_path)
    if cached is not None:
        return cached

    # Cache miss (absent, empty, or truncated/corrupt): rebuild from raw files.
    airfoils_db, complete = _build_db_from_dat_files(data_folder)

    # Persist atomically so subsequent/parallel runs reuse a complete pickle.
    # Skip persisting a partial build (some .dat file was unreadable) so the
    # cache is never poisoned with an incomplete database.
    if airfoils_db and complete:
        _atomic_pickle_dump(airfoils_db, pickle_file_path)

    return airfoils_db

def get_baselines(data_folder: str, expected_names: list[str]) -> list[Airfoil]:
    """Loads exact specified baselines from the database."""
    global _CACHED_AIRFOIL_DB_DICT
    
    if _CACHED_AIRFOIL_DB_DICT is None:
        full_db = load_airfoil_database_from_files(data_folder)
        _CACHED_AIRFOIL_DB_DICT = {af.name: af for af in full_db}
        
    baselines = []
    for name in expected_names:
        if name in _CACHED_AIRFOIL_DB_DICT:
            baselines.append(_CACHED_AIRFOIL_DB_DICT[name])
        else:
            raise RuntimeError(f"Requested baseline '{name}' not found in the airfoil database.")
            
    return baselines

# =============================================================================
# GEOMETRY CORRECTION & MORPHING
# =============================================================================
def correct_airfoil_geometry(airfoil_to_correct: Airfoil) -> Airfoil:
    if Polygon is None:
        return airfoil_to_correct

    points = list(zip(X_INTERP, airfoil_to_correct.colloc_vec))
    if tuple(points[0]) != tuple(points[-1]):
        points.append(points[0])

    try:
        airfoil_shape_current = Polygon(points).simplify(1e-5, preserve_topology=True)
    except Exception:
        return airfoil_to_correct

    try:
        airfoil_shape_current = make_valid(airfoil_shape_current)
        if not isinstance(airfoil_shape_current, Polygon):
            if hasattr(airfoil_shape_current, 'geoms'):
                polygons = [g for g in airfoil_shape_current.geoms if isinstance(g, Polygon)]
                if polygons: airfoil_shape_current = max(polygons, key=lambda p: p.area)
                else: return airfoil_to_correct
            else: return airfoil_to_correct

        # Must match the active collocation distribution (cosine) to avoid shape distortion.
        x_slices = X_INTERP[NUM_POINTS_INTERP // 2:]
        upper_surface_pts, lower_surface_pts = [], []
        final_boundary = LineString(airfoil_shape_current.exterior.coords)

        for x_val in x_slices:
            vertical_line = LineString([(x_val, -1), (x_val, 1)])
            intersection = final_boundary.intersection(vertical_line)
            if intersection.is_empty: continue
            
            y_at_x = []
            if isinstance(intersection, Point): y_at_x.append(intersection.y)
            elif hasattr(intersection, 'geoms'):
                for geom_item in intersection.geoms:
                    if isinstance(geom_item, Point): y_at_x.append(geom_item.y)

            if not y_at_x: continue
            y_at_x.sort(reverse=True)
            upper_surface_pts.append((x_val, y_at_x[0]))
            lower_surface_pts.append((x_val, y_at_x[-1]))
        
        if not upper_surface_pts or not lower_surface_pts: return airfoil_to_correct

        x_upper_hit = np.array([p[0] for p in upper_surface_pts])
        y_upper_hit = np.array([p[1] for p in upper_surface_pts])
        x_lower_hit = np.array([p[0] for p in lower_surface_pts])
        y_lower_hit = np.array([p[1] for p in lower_surface_pts])

        y_upper_resampled = _resample_surface(x_upper_hit, y_upper_hit, x_slices)
        y_lower_resampled = _resample_surface(x_lower_hit, y_lower_hit, x_slices)
        if y_upper_resampled is None or y_lower_resampled is None:
            return airfoil_to_correct

        y_upper_smoothed = _smooth_surface_preserve_endpoints(y_upper_resampled, window_size=5)
        y_lower_smoothed = _smooth_surface_preserve_endpoints(y_lower_resampled, window_size=5)

        # Enforce a unique LE and two explicit TE points (upper/lower branch endpoints).
        y_le = 0.5 * (y_upper_smoothed[0] + y_lower_smoothed[0])
        y_upper_smoothed[0] = y_le
        y_lower_smoothed[0] = y_le

        # Keep a small positive interior thickness to avoid local crossings.
        y_upper_smoothed, y_lower_smoothed = _enforce_min_interior_thickness(
            y_upper_smoothed,
            y_lower_smoothed,
            min_thickness=MIN_INTERIOR_THICKNESS,
        )

        # Prevent self-intersection at trailing edge by enforcing TE ordering.
        te_thickness = y_upper_smoothed[-1] - y_lower_smoothed[-1]
        if te_thickness < MIN_TRAILING_EDGE_THICKNESS:
            te_delta = 0.5 * (MIN_TRAILING_EDGE_THICKNESS - te_thickness)
            y_upper_smoothed[-1] += te_delta
            y_lower_smoothed[-1] -= te_delta

        y_selig_final = np.concatenate((y_upper_smoothed[::-1], y_lower_smoothed[1:]))

        return Airfoil(
            airfoil_id=airfoil_to_correct.airfoil_id,
            name=f"{airfoil_to_correct.name}_corrected",
            colloc_vec=y_selig_final,
            x_raw=X_INTERP,
            y_raw=y_selig_final.copy()
        )
    except Exception:
        return airfoil_to_correct

def create_morphed_airfoil(weights: np.ndarray, baseline_airfoils: list[Airfoil], correct_geometry: bool = True) -> Airfoil:
    if len(weights) != len(baseline_airfoils):
        raise ValueError("Length of weights must match the number of baseline airfoils.")

    weights_np = np.array(weights)
    
    # Intentionally bypass original DbM L1 normalization here. 
    # Normalization should be explicitly requested via TestAirfoils API args.
    
    if np.allclose(weights_np, 0):
        return Airfoil(airfoil_id=-1, name='Morphed_FlatLine', colloc_vec=np.zeros_like(X_INTERP),
                       x_raw=X_INTERP.copy(), y_raw=np.zeros_like(X_INTERP))

    y_morphed_sum = np.zeros_like(X_INTERP)
    for airfoil, weight in zip(baseline_airfoils, weights_np):
        y_morphed_sum += weight * airfoil.get_interpolated_data()
    
    weight_str = "_".join(f"{w:.4f}" for w in weights_np)
    morphed_name = f"Morphed_W[{weight_str[:50]}]"

    morphed_obj = Airfoil(airfoil_id=-1, name=morphed_name, colloc_vec=y_morphed_sum,
                          x_raw=X_INTERP.copy(), y_raw=y_morphed_sum.copy())
    
    if correct_geometry:
        return correct_airfoil_geometry(morphed_obj)
    return morphed_obj

def _evaluate_single_candidate(
    phi: np.ndarray,
    weight_range: tuple[float, float],
    normalization: str | None,
    data_folder: str,
    selected_baseline_names: list[str],
    xfoil_config: dict,
    m : int,
) -> Airfoil:
    """Worker-safe single-candidate evaluator for parallel TestAirfoils execution."""
    baselines = get_baselines(data_folder, selected_baseline_names)
    w_lower, w_upper = weight_range

    weights = w_lower + phi * (w_upper - w_lower)

    if normalization == 'SUM':
        norm = np.sum(weights)
        if norm > 1e-8:
            weights = weights / norm
        else:
            raise ValueError("Sum of weights is too close to zero for normalization.")
    elif normalization == 'ABS_SUM':
        norm = np.sum(np.abs(weights))
        if norm > 1e-8:
            weights = weights / norm
        else:
            raise ValueError("Sum of absolute weights is too close to zero for normalization.")

    morphed_airfoil = create_morphed_airfoil(weights, baselines, correct_geometry=True)
    # Format weights to 2 decimals and limit name to 70 characters (Fortran limit)
    morphed_airfoil.name = "_".join(f"{w:.2f}" for w in weights)[:70]

    if xfoil_config.get('xfoil_evaluation', False):
        strict = bool(xfoil_config.get('xfoil_strict', True))
        # Give transient no-parse runs another chance. In heavily parallel runs,
        # XFOIL can occasionally deadlock or stall I/O and produce empty polar files
        # even when the same design converges in other attempts.
        retry_count = max(0, int(xfoil_config.get('xfoil_retry', 1)))

        def _is_empty_polar(metrics: dict | None) -> bool:
            if not metrics:
                return True
            return (
                len(metrics.get('alpha', [])) == 0
                and len(metrics.get('cl', [])) == 0
                and len(metrics.get('cd', [])) == 0
            )

        last_exc = None
        for attempt in range(retry_count + 1):
            try:
                metrics = run_xfoil_evaluation(morphed_airfoil, xfoil_config, m)
                # Retry not only on an empty polar but also when any alpha-scan stage was killed by
                # the wall-clock timeout. A lost stage leaves a NON-empty (coarse-only) polar, so the
                # empty-polar test alone would publish a value scored from fewer stages than the
                # protocol specifies -- i.e. a load-dependent objective.
                incomplete = _is_empty_polar(metrics) or int(metrics.get('scan_timeouts', 0) or 0) > 0
                if incomplete and attempt < retry_count:
                    continue
                morphed_airfoil.xfoil_result = metrics
                break
            except Exception as exc:
                last_exc = exc
                if attempt < retry_count:
                    continue
                if strict:
                    raise
                morphed_airfoil.xfoil_result = {
                    'error': str(exc),
                    'runner': None,
                }

        if morphed_airfoil.xfoil_result is None and last_exc is not None and not strict:
            morphed_airfoil.xfoil_result = {
                'error': str(last_exc),
                'runner': None,
            }

    return morphed_airfoil

class AirfoilEvaluationResult:
    """Container for a generated airfoil and optional XFOIL-derived outputs."""

    def __init__(
        self,
        airfoil: Airfoil,
        xfoil_result: dict | None = None,
        objectives: float | list[float] | None = None,
    ):
        self.airfoil = airfoil
        self.xfoil_result = xfoil_result
        self.objectives = objectives


def _extract_objectives_from_xfoil_result(xfoil_result: dict, m: int) -> float | list[float]:
    if m not in (1, 2):
        raise ValueError("When xfoil_evaluation=True, only m=1 or m=2 are currently supported.")

    cl_cd_max_raw = xfoil_result.get('cl_cd_max', np.nan)
    delta_alpha_raw = xfoil_result.get('delta_alpha', np.nan)

    cl_cd_max = np.nan if cl_cd_max_raw is None else float(cl_cd_max_raw)
    delta_alpha = np.nan if delta_alpha_raw is None else float(delta_alpha_raw)

    # Defense-in-depth: the stall margin is physically non-negative, so enforce
    # that guarantee again at the objective boundary. This shields the objective
    # actually consumed by an optimizer from any upstream numerical noise (or a
    # stale/foreign metrics dict) that might carry a small negative value.
    if not np.isnan(delta_alpha):
        delta_alpha = delta_alpha if delta_alpha > 0.0 else 0.0

    if m == 1:
        return cl_cd_max
    return [cl_cd_max, delta_alpha]


def _format_testairfoils_output(
    airfoils: list[Airfoil],
    xfoil_evaluation: bool,
    m: int,
) -> list[AirfoilEvaluationResult]:
    """
    Always return container objects so callers can access generated airfoils.
    When XFOIL is enabled, objective values are populated from attached XFOIL results.
    """
    outputs: list[AirfoilEvaluationResult] = []
    for airfoil in airfoils:
        xr = airfoil.xfoil_result if xfoil_evaluation else None
        objectives = None
        if xfoil_evaluation:
            objectives = _extract_objectives_from_xfoil_result(xr or {}, m)

        outputs.append(AirfoilEvaluationResult(airfoil=airfoil, xfoil_result=xr, objectives=objectives))

    return outputs

# =============================================================================
# MAIN API: TestAirfoils
# =============================================================================
def TestAirfoils(x: np.ndarray, args: dict = None, m: int = 2) -> list[AirfoilEvaluationResult]:
    """
    Test function for generating DbM airfoils based on an N x D candidate matrix.
    
        Parameters:
        - x: N x D array, where N is candidates and D is design parameters.
                 Input values must range from 0 to 1.
        - args: Configuration dictionary supporting:
            AIRFOIL DESIGN ARGS:
                - 'airfoil_db_dir': Path to airfoil database (defaults to DATA_FOLDER).
                - 'dbm_baselines': List of airfoil names to use as baselines (defaults to EXPECTED_BASELINES).
                - 'dbm_weight_range': [lower, upper] limits (default: [-1.0, 1.0]). Allows negative weights for extrapolative morphing.
                - 'dbm_normalization': None (default), 'SUM', or 'ABS_SUM'.
            AIRFOIL EVALUATION ARGS:
                - 'xfoil_evaluation': Run XFOIL evaluation after geometry generation (default: True).
                - 'xfoil_backend': 'apptainer' (default), 'auto', or 'native'.
                    ('auto': apptainer image first, then system 'xfoil').
                - 'apptainer_image': Path to apptainer image containing xfoil (default: XFOIL_APP).
                - 'xfoil_iter': Max XFOIL iterations per alpha (default: 200).
                - 'xfoil_timeout': Timeout in seconds per candidate (default: 60.0).
                - 'xfoil_retry': Number of reattempts when no polar points are parsed at all (default: 1).
                    Useful for transient XFOIL deadlock/no-parse cases that can occur in multiprocessing,
                    even when the design itself converges in other runs.
                - 'xfoil_strict': If True, fail on XFOIL errors; if False, attach error in airfoil.xfoil_result.
                - 'alfa_start', 'alfa_end': Polar angle range (defaults: 0, 45).
                - 'reynolds': Reynolds number (defaults to REYNOLDS).
                - 'mach': Mach number for XFOIL (default: MACH=0).
                - 'n_crit': e^N critical amplification factor (default: N_CRIT=9).
                - 'repanel_n': Internal XFOIL repanel node count via PPAR/N (default: 160).
            MULTIPROCESSING ARGS:
                - 'parallel': Enable multiprocessing across candidates (default: True).
                - 'max_workers': Max worker processes (default: total CPU count in environment).
    - m: Number of objectives (1 for Cl/Cd <single objective>, 2 for Cl/Cd and delta alpha <bi-objective>).
    
    Returns:
    - A list of AirfoilEvaluationResult objects of length N.
      Each element provides:
        - .airfoil: generated Airfoil object (always available)
        - .xfoil_result: raw XFOIL metrics dict when xfoil_evaluation=True, else None
        - .objectives:
            - None when xfoil_evaluation=False
            - m=1 -> Cl/Cd_max (float)
            - m=2 -> [Cl/Cd_max, delta_alpha]
    """
    if args is None:
        args = {}
        
    # Extract configuration with safe defaults
    data_folder = args.get('airfoil_db_dir', DATA_FOLDER)
    weight_range = args.get('dbm_weight_range', [-1.0, 1.0])
    normalization = args.get('dbm_normalization', None)
    expected_baselines = args.get('dbm_baselines', EXPECTED_BASELINES)
    parallel = args.get('parallel', True)
    # CPUs this process may actually use. os.cpu_count() reports the whole node, which
    # oversubscribes a shared-partition allocation (e.g. 128 workers on a 32-core request) and
    # makes every XFOIL process contend, inflating wall-clock and timeout risk.
    try:
        cpu_total = max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        cpu_total = max(1, os.cpu_count() or 1)
    max_workers_cfg = args.get('max_workers', cpu_total)
    max_workers = min(max(1, int(max_workers_cfg)), cpu_total)
    xfoil_config = {
        'xfoil_evaluation': args.get('xfoil_evaluation', True),
        'xfoil_backend': args.get('xfoil_backend', 'apptainer'),
        'apptainer_image': args.get('apptainer_image', XFOIL_APP),
        'reynolds': args.get('reynolds', REYNOLDS),
        'mach': args.get('mach', MACH),
        'n_crit': args.get('n_crit', N_CRIT),
        'alfa_start': args.get('alfa_start', 0.0),
        'alfa_end': args.get('alfa_end', 45.0),
        'repanel_n': args.get('repanel_n', 160),
        'xfoil_iter': args.get('xfoil_iter', 200),
        'xfoil_timeout': args.get('xfoil_timeout', 60.0),
        'xfoil_retry': args.get('xfoil_retry', 1),
        'xfoil_strict': args.get('xfoil_strict', True),
        # The nonphysical-efficiency guard: a HARD ceiling of 350 at every Reynolds number. This
        # is part of the objective definition, not a diagnostic -- the released objective is
        # min(XFOIL (Cl/Cd)_max, 350), applied uniformly and deterministically. It must be
        # forwarded through this dict; omitting the key (as an earlier version did) meant a
        # caller's value was silently discarded.
        'clcd_ceiling': args.get('clcd_ceiling', 350.0),
    }
    
    # Ensure standard 2D array matrix behavior
    x = np.atleast_2d(x)

    N, D = x.shape
    
    # Validate D against the length of expected baselines
    if D > len(expected_baselines):
        raise ValueError(f"Number of design parameters (D={D}) exceeds the number of provided baselines ({len(expected_baselines)}).")
    
    # Slice exactly D baselines from the provided expected baselines list
    selected_baseline_names = expected_baselines[:D]

    # WEIGHT RANGE VALIDATION ---
    w_lower, w_upper = float(weight_range[0]), float(weight_range[1])
    if w_lower > 0.0 or w_upper < 1.0:
        raise ValueError(
            f"Invalid dbm_weight_range: [{w_lower}, {w_upper}]. "
            "The range must fully encompass [0.0, 1.0] to guarantee that "
            "pure baseline reconstruction remains mathematically possible."
        )
    weight_range_tuple = (w_lower, w_upper)
    # -------------------------------------------
    xfoil_evaluation = bool(xfoil_config.get('xfoil_evaluation', True))

    if parallel and N > 1 and max_workers > 1:
        workers = min(max_workers, N)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            airfoils = list(
                executor.map(
                    _evaluate_single_candidate,
                    x,
                    repeat(weight_range_tuple),
                    repeat(normalization),
                    repeat(data_folder),
                    repeat(selected_baseline_names),
                    repeat(xfoil_config),
                    repeat(m),
                )
            )
        return _format_testairfoils_output(airfoils, xfoil_evaluation=xfoil_evaluation, m=m)

    # Serial fallback (also useful for very small N).
    airfoils = [
        _evaluate_single_candidate(
            phi=x[i, :],
            weight_range=weight_range_tuple,
            normalization=normalization,
            data_folder=data_folder,
            selected_baseline_names=selected_baseline_names,
            xfoil_config=xfoil_config,
            m=m,
        )
        for i in range(N)
    ]
    return _format_testairfoils_output(airfoils, xfoil_evaluation=xfoil_evaluation, m=m)

def VerifyDesigns(
    x: np.ndarray,
    args: dict = None,
    m: int = 2,
    eps: float = 1e-6,
    n_dir: int = 4,
    rel_tol: float = 0.02,
    objective: int = 0,
    directions: str = "axes",
    seed: int = 0,
) -> list[dict]:
    """Perturbation test for designs that are about to be published as reference solutions.

    An objective returned by this interface is a deterministic function of the design vector, but
    determinism is not stability: XFOIL can hold two different boundary-layer solutions for
    geometrically near-identical sections (a long laminar run versus a transitioned one), so an
    isolated design vector can score far above every neighbor. Such a point is reproducible yet
    unreachable by search, and it has no business defining a reference optimum or a Pareto front.

    The verdict compares the design against the MEDIAN of its neighbors, not against the worst of
    them. A design sitting on a ridge has one direction that falls away and the rest that agree; a
    design that is truly isolated has a neighborhood that agrees with itself and not with the
    design. Only the second should be rejected, and only the median separates the two.

    The verdict is taken on ONE objective (default the first). Objectives differ in how sharply the
    protocol resolves them -- a stall margin is the gap between two features located on a smoothed
    alpha scan, so it moves under perturbations that leave the efficiency untouched -- and folding
    such an objective into a pass/fail rule rejects the extremes an optimizer is supposed to find.
    Deviations for every objective are reported either way, so a caller can characterize the rest.

    This check is intentionally NOT part of TestAirfoils: at the default directions='axes' it costs
    (1 + 2D) evaluations per design, and (1 + n_dir) under directions='random', so running it inside
    the objective would multiply the cost of an optimization run.
    It is meant for the O(front size) candidate reference set at the END of a study -- a fraction
    of a percent of a campaign's budget -- rather than for the O(budget) hot path.

        Parameters:
        - x: N x D array of designs to verify (same convention as TestAirfoils).
        - args, m: passed through to TestAirfoils unchanged.
        - eps: perturbation radius in design space (L2), applied in n_dir random directions.
        - n_dir: number of perturbed neighbors per design.
        - rel_tol: a design is 'robust' when the median neighbor's value of the deciding objective
          is within this relative deviation of the design's own value.
        - objective: index of the deciding objective (0 -> (Cl/Cd)_max).
        - directions: 'axes' (default) probes x +/- eps e_i on every coordinate, a canonical set of
          2D neighbors with no random choice in it, so a verdict carries no seed. 'random' draws n_dir
          directions instead, seeded per design. Measured on the designs this benchmark rejected, a
          random draw of 4 or 8 directions gives a verdict that flips between draws for 6 of 11
          borderline designs -- the neighborhood is truly heterogeneous, and no amount of sampling
          removes the arbitrariness -- which is why the canonical set is the default.
        - seed: fixes the directions when directions='random'; unused for 'axes'.

    Returns a list of N dicts with 'objectives', 'neighbors', 'median_neighbor', 'rel_dev' (one
    entry per objective), 'n_informative' and 'robust'. A neighbor that comes back at the failure
    floor is counted as uninformative rather than as disagreement: it says the neighbor could not be
    evaluated, not that the design's own value is isolated. Designs and their neighbors go out in
    ONE batched TestAirfoils call, so the check runs at the interface's full parallel throughput.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    N, D = x.shape
    if n_dir < 1:
        raise ValueError("n_dir must be at least 1.")
    if not 0 <= objective < m:
        raise ValueError(f"objective index {objective} outside the {m} objectives.")
    if directions == "axes":
        # Canonical set: x +/- eps e_i on every coordinate. No seed, no draw, so the verdict is a
        # function of the design vector and the tolerances alone.
        unit = np.concatenate([np.eye(D), -np.eye(D)], axis=0)      # (2D, D)
        dirs = np.broadcast_to(unit, (N, 2 * D, D))
        n_probe = 2 * D
    elif directions == "random":
        # Directions derived PER DESIGN from the bytes of its own coordinates, not from a batch-level
        # generator: with a shared generator the directions a design receives depend on how many
        # designs were in the call and in what order, so the same design could be verified against
        # different neighbors in different runs.
        def _dirs(xi):
            h = hashlib.sha256(np.ascontiguousarray(xi, dtype=np.float64).tobytes()
                               + str(int(seed)).encode()).digest()
            g = np.random.default_rng(int.from_bytes(h[:8], "little"))
            d = g.normal(size=(n_dir, len(xi)))
            return d / np.linalg.norm(d, axis=1, keepdims=True)

        dirs = np.stack([_dirs(xi) for xi in x])
        n_probe = n_dir
    else:
        raise ValueError("directions must be 'axes' or 'random'")
    batch = [x]
    for k in range(n_probe):
        batch.append(np.clip(x + eps * dirs[:, k, :], 0.0, 1.0))
    Y = np.array([
        np.atleast_1d(np.asarray(r.objectives, dtype=float))
        for r in TestAirfoils(np.vstack(batch), args=args, m=m)
    ]).reshape(n_probe + 1, N, -1)

    out = []
    for i in range(N):
        y0, yn = Y[0, i], Y[1:, i]
        keep = yn[:, objective] > 0.0
        info = yn[keep]
        med = np.median(info, axis=0) if len(info) else np.full_like(y0, np.nan)
        dev = (np.abs(med - y0) / np.maximum(np.abs(y0), 1e-12) if len(info)
               else np.full_like(y0, np.nan))
        # the spread of the informative neighbors on the deciding objective, so the verdict can be
        # re-thresholded from the released record without re-evaluating anything
        q = (np.percentile(np.abs(info[:, objective] - y0[objective])
                           / max(abs(y0[objective]), 1e-12), [0, 25, 50, 75, 100]).tolist()
             if len(info) else [float('nan')] * 5)
        out.append({
            'objectives': y0.tolist() if m > 1 else float(y0[0]),
            'neighbors': yn.tolist(),
            'median_neighbor': med.tolist(),
            'rel_dev': dev.tolist(),
            'abs_dev': (np.abs(med - y0)).tolist(),
            'rel_dev_quantiles': q,
            'frac_within_tol': (float(np.mean(np.abs(info[:, objective] - y0[objective])
                                              <= rel_tol * max(abs(y0[objective]), 1e-12)))
                                if len(info) else float('nan')),
            'directions': directions,
            'n_probes': int(n_probe),
            'n_informative': int(keep.sum()),
            'robust': bool(keep.sum() >= 2 and dev[objective] <= rel_tol),
        })
    return out
