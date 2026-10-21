"""Closed 2-D path sampled at (nearly) uniform arc length: shared geometry code
for the track centreline and for racing lines."""
from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class ClosedPath:
    """A closed polyline with per-sample arc length, heading and curvature."""

    def __init__(self, points: np.ndarray, spacing: float | None = None):
        """``points`` must be (nearly) uniformly spaced; ``spacing`` is ignored and
        kept only for API symmetry (the uniform spacing length / n is used)."""
        self.points = np.asarray(points, float)
        self.n = len(self.points)
        seg = np.linalg.norm(np.roll(self.points, -1, axis=0) - self.points, axis=1)
        self.length = float(seg.sum())
        self.spacing = self.length / self.n
        self.s = np.arange(self.n) * self.spacing
        d = (np.roll(self.points, -1, axis=0) - np.roll(self.points, 1, axis=0)) / 2.0
        self.heading = np.arctan2(d[:, 1], d[:, 0])
        ds = np.linalg.norm(d, axis=1)
        dh = wrap_angle(np.roll(self.heading, -1) - np.roll(self.heading, 1)) / 2.0
        self.curvature = dh / ds
        self.tangent = np.stack([np.cos(self.heading), np.sin(self.heading)], axis=1)
        self.normal = np.stack([-np.sin(self.heading), np.cos(self.heading)], axis=1)

    # ---------------------------------------------------------------- lookups
    def _frac_index(self, s: float):
        q = (s % self.length) / self.spacing
        i0 = int(np.floor(q)) % self.n
        return i0, (i0 + 1) % self.n, q - np.floor(q)

    def point_at(self, s: float) -> np.ndarray:
        i0, i1, w = self._frac_index(s)
        return (1 - w) * self.points[i0] + w * self.points[i1]

    def heading_at(self, s: float) -> float:
        i0, i1, w = self._frac_index(s)
        return float(self.heading[i0] + w * wrap_angle(self.heading[i1] - self.heading[i0]))

    def curvature_ahead(self, s: float, distances) -> np.ndarray:
        """Curvature at s + distances (linear interpolation, periodic)."""
        q = ((np.asarray(distances, float) + s) % self.length) / self.spacing
        i0 = np.floor(q).astype(int) % self.n
        i1 = (i0 + 1) % self.n
        w = q - np.floor(q)
        return (1 - w) * self.curvature[i0] + w * self.curvature[i1]

    def value_at(self, values: np.ndarray, s: float) -> float:
        """Interpolate an arbitrary per-sample array at arc length s."""
        i0, i1, w = self._frac_index(s)
        return float((1 - w) * values[i0] + w * values[i1])

    def nearest_index(self, x: float, y: float, hint: int | None = None, window: int = 25) -> int:
        if hint is None:
            d2 = np.sum((self.points - (x, y)) ** 2, axis=1)
            return int(np.argmin(d2))
        idx = (hint + np.arange(-window, window + 1)) % self.n
        d2 = np.sum((self.points[idx] - (x, y)) ** 2, axis=1)
        return int(idx[np.argmin(d2)])

    def localize(self, x: float, y: float, hint: int | None = None):
        """Project (x, y) onto the path -> (s, lateral offset (+ = left), nearest index)."""
        i = self.nearest_index(x, y, hint)
        p = self.points[i]
        rel = np.array([x, y]) - p
        along = float(rel @ self.tangent[i])
        if along >= 0:
            j = (i + 1) % self.n
            a, seg = p, self.points[j] - p
            s_base = self.s[i]
        else:
            j = (i - 1) % self.n
            a, seg = self.points[j], p - self.points[j]
            s_base = self.s[j]
        seg_len = float(np.linalg.norm(seg))
        u = float(np.clip(((np.array([x, y]) - a) @ seg) / (seg_len ** 2), 0.0, 1.0))
        foot = a + u * seg
        tn = seg / seg_len
        s = (s_base + u * seg_len) % self.length
        dx, dy = x - foot[0], y - foot[1]
        lat = float(tn[0] * dy - tn[1] * dx)
        return float(s), lat, i

    def resample(self, spacing: float) -> np.ndarray:
        """Return points re-sampled at the given arc-length spacing."""
        q = np.vstack([self.points, self.points[:1]])
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = int(round(s[-1] / spacing))
        s_new = np.linspace(0.0, s[-1], n, endpoint=False)
        return np.stack([np.interp(s_new, s, q[:, i]) for i in range(2)], axis=1)
