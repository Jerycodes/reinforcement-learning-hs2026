"""Racing line: minimum-curvature path inside the track limits plus a
friction-circle speed profile.

The path is parametrised as lateral offsets ``alpha_i`` from a reference line
along its left normals.  Curvature is linearised around the reference, which
turns "minimise the sum of squared curvatures subject to staying inside the
track" into a bounded least-squares problem (Heilmeier et al., 2019).  A few
re-linearisation iterations converge to the minimum-curvature line.

The speed profile is the classic forward/backward pass with a friction circle:
lateral limit from curvature, longitudinal limits from engine power, brakes
and the remaining grip.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear

from .kart import G, KartParams
from .path import ClosedPath
from .track import DATA_DIR, Track


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

def ray_to_polyline(origin: np.ndarray, direction: np.ndarray, pts: np.ndarray, idx: int, window: int = 12,
                    closed: bool = True) -> float:
    """Distance from ``origin`` along ``direction`` to a polyline, searching the
    segments around sample ``idx``. Returns inf when nothing is hit."""
    n = len(pts)
    ids = (idx + np.arange(-window, window + 1)) % n if closed else np.clip(idx + np.arange(-window, window + 1), 0, n - 1)
    a = pts[ids]
    b = pts[(ids + 1) % n]
    e = b - a
    # solve origin + t*d = a + u*e  ->  [d, -e] [t, u]^T = a - origin
    det = direction[0] * (-e[:, 1]) - direction[1] * (-e[:, 0])
    rhs = a - origin
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (rhs[:, 0] * (-e[:, 1]) - rhs[:, 1] * (-e[:, 0])) / det
        u = (direction[0] * rhs[:, 1] - direction[1] * rhs[:, 0]) / det
    ok = np.isfinite(t) & (t > 0) & (u >= 0) & (u <= 1)
    return float(t[ok].min()) if ok.any() else float("inf")


# --------------------------------------------------------------------------- #
# minimum-curvature optimisation
# --------------------------------------------------------------------------- #

def minimum_curvature_line(track: Track, margin: float = 1.0, iterations: int = 4,
                           spacing: float = 1.0, verbose: bool = False) -> np.ndarray:
    """Return the minimum-curvature line (N, 2) that stays ``margin`` metres inside the edges."""
    ref = ClosedPath(track.centerline)
    for it in range(iterations):
        n = ref.n
        pts, nrm = ref.points, ref.normal
        # corridor along the normal of each reference point
        hi = np.empty(n)
        lo = np.empty(n)
        for i in range(n):
            j = track.nearest_index(pts[i, 0], pts[i, 1])
            dl = ray_to_polyline(pts[i], nrm[i], track.left_edge, j)
            dr = ray_to_polyline(pts[i], -nrm[i], track.right_edge, j)
            hi[i] = dl - margin
            lo[i] = -(dr - margin)
            if hi[i] < lo[i]:  # corridor narrower than 2*margin: stay in the middle
                mid = (dl - dr) / 2.0
                hi[i] = lo[i] = mid
        # linearised curvature: kappa_i*h^2 = d_i + A_i alpha
        im1, ip1 = np.roll(np.arange(n), 1), np.roll(np.arange(n), -1)
        d = np.einsum("ij,ij->i", pts[im1] - 2 * pts + pts[ip1], nrm)
        rows = np.concatenate([np.arange(n)] * 3)
        cols = np.concatenate([im1, np.arange(n), ip1])
        vals = np.concatenate([np.einsum("ij,ij->i", nrm[im1], nrm), np.full(n, -2.0),
                               np.einsum("ij,ij->i", nrm[ip1], nrm)])
        A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        res = lsq_linear(A, -d, bounds=(lo - 1e-9, hi + 1e-9), method="trf", tol=1e-10,
                         lsmr_tol="auto", max_iter=5000)
        alpha = res.x
        new_pts = pts + alpha[:, None] * nrm
        ref = ClosedPath(ClosedPath(new_pts).resample(spacing))
        if verbose:
            print(f"  iteration {it + 1}: length {ref.length:.1f} m, "
                  f"max |curvature| {np.abs(ref.curvature).max():.3f} 1/m, "
                  f"mean |alpha| {np.abs(alpha).mean():.2f} m")
    return ref.points


# --------------------------------------------------------------------------- #
# speed profile
# --------------------------------------------------------------------------- #

def speed_profile(path: ClosedPath, kart: KartParams | None = None, grip_scale: float = 1.0,
                  safety: float = 0.92, v_top: float | None = None, brake_share: float = 0.75) -> np.ndarray:
    """Quasi-steady-state speed [m/s] per sample from a friction-circle model.

    ``brake_share``: fraction of the brake force assumed usable (rental karts
    brake with the rear axle only, which also has to provide cornering grip).
    """
    p = kart or KartParams()
    mu = p.mu * grip_scale
    a_max = mu * G * safety
    h = path.spacing
    n = path.n
    kappa = np.abs(path.curvature)
    v_top = v_top if v_top is not None else (p.speed_limit if p.speed_limit > 0 else 30.0)
    v = np.minimum(np.sqrt(a_max / np.maximum(kappa, 1e-4)), v_top)

    def a_long_avail(vi, ki):
        a_lat = vi * vi * ki
        return np.sqrt(max(a_max * a_max - a_lat * a_lat, 0.0))

    def a_engine(vi):
        f = min(p.drive_force_max, p.power_max / max(vi, 1.0))
        return (f - p.drag_coeff * vi * vi - p.roll_resistance * p.mass * G) / p.mass

    def a_brake(vi):
        return (brake_share * p.brake_force_max + p.drag_coeff * vi * vi + p.roll_resistance * p.mass * G) / p.mass

    # forward pass (twice around for periodicity), starting at the slowest point
    start = int(np.argmin(v))
    for k in range(2 * n):
        i = (start + k) % n
        j = (i + 1) % n
        a = min(a_engine(v[i]), a_long_avail(v[i], kappa[i]))
        v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * max(a, 0.0) * h))
    # backward pass (twice around)
    for k in range(2 * n):
        j = (start - k) % n
        i = (j - 1) % n
        a = min(a_brake(v[j]), a_long_avail(v[j], kappa[j]))
        v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * max(a, 0.0) * h))
    return v


def lap_time_from_profile(path: ClosedPath, v: np.ndarray) -> float:
    return float(np.sum(path.spacing / np.maximum(v, 0.1)))


# --------------------------------------------------------------------------- #
# RaceLine container
# --------------------------------------------------------------------------- #

class RaceLine(ClosedPath):
    """A closed reference line with a speed profile."""

    def __init__(self, points: np.ndarray, speed: np.ndarray | None = None, meta: dict | None = None):
        super().__init__(points)
        self.speed = np.asarray(speed, float) if speed is not None else None
        self.meta = meta or {}

    @property
    def lap_time_estimate(self) -> float | None:
        return lap_time_from_profile(self, self.speed) if self.speed is not None else None

    def save(self, path: str | Path):
        d = {
            "points": self.points.round(3).tolist(),
            "s": self.s.round(3).tolist(),
            "curvature": self.curvature.round(5).tolist(),
            "speed": self.speed.round(3).tolist() if self.speed is not None else None,
            "length_m": round(self.length, 3),
            "lap_time_estimate_s": round(self.lap_time_estimate, 3) if self.speed is not None else None,
            "meta": self.meta,
        }
        Path(path).write_text(json.dumps(d))

    @classmethod
    def load(cls, name_or_path: str | Path = "race_inn_roggwil_raceline") -> "RaceLine":
        p = Path(name_or_path)
        if not p.exists():
            p = DATA_DIR / f"{name_or_path}.json"
        d = json.loads(p.read_text())
        return cls(np.asarray(d["points"], float), d.get("speed"), d.get("meta"))

    @classmethod
    def from_track_centerline(cls, track: Track, kart: KartParams | None = None) -> "RaceLine":
        line = cls(track.centerline)
        line.speed = speed_profile(line, kart)
        line.meta = {"source": "centerline"}
        return line

    @classmethod
    def compute(cls, track: Track, kart: KartParams | None = None, margin: float = 1.0,
                iterations: int = 4, safety: float = 0.92, verbose: bool = False) -> "RaceLine":
        pts = minimum_curvature_line(track, margin=margin, iterations=iterations, verbose=verbose)
        line = cls(pts)
        line.speed = speed_profile(line, kart, safety=safety)
        line.meta = {"source": "minimum_curvature", "margin_m": margin, "iterations": iterations,
                     "safety": safety, "kart": (kart or KartParams()).to_dict()}
        return line
