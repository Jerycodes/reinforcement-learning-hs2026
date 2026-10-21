"""Track geometry: centreline, edges, progress along the lap, drivable-area grid,
lidar ray casting.

All quantities are in metres / radians in the world frame (x east, y north).
The track data is produced by ``tools/extract_track.py`` (see docs/strecke.md).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from .path import ClosedPath, wrap_angle

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class PitLane:
    centerline: np.ndarray  # (M, 2)
    heading: np.ndarray
    width: np.ndarray
    left_edge: np.ndarray
    right_edge: np.ndarray
    length: float
    entry_index: int
    exit_index: int
    entry_s: float
    exit_s: float
    bypassed_length: float


class Track(ClosedPath):
    """Closed-loop track with 1 m samples, plus a raster of the drivable area.

    Raster values (``self.grid``): 0 = off track, 1 = main track, 2 = pit lane.
    ``self.dist_grid`` holds the distance [m] to the nearest off-track cell for
    every cell inside the main track (0 outside).
    """

    MAIN, PIT = 1, 2

    def __init__(self, name: str = "race_inn_roggwil", data_dir: Path | None = None):
        data_dir = Path(data_dir) if data_dir else DATA_DIR
        d = json.loads((data_dir / f"{name}.json").read_text())
        self.name: str = d["name"]
        self.length: float = float(d["length_m"])
        self.spacing: float = float(d["sample_spacing_m"])
        super().__init__(np.asarray(d["centerline"], float), spacing=self.spacing)
        self.centerline = self.points
        self.width = np.asarray(d["width"], float)
        self.left_edge = np.asarray(d["left_edge"], float)
        self.right_edge = np.asarray(d["right_edge"], float)
        self.half_width_left = np.linalg.norm(self.left_edge - self.centerline, axis=1)
        self.half_width_right = np.linalg.norm(self.right_edge - self.centerline, axis=1)
        self.direction: str = d.get("direction", "")
        b = d["bounds"]
        self.bounds = (b["x_min"], b["x_max"], b["y_min"], b["y_max"])
        p = d["pit_lane"]
        self.pit = PitLane(
            centerline=np.asarray(p["centerline"], float), heading=np.asarray(p["heading"], float),
            width=np.asarray(p["width"], float), left_edge=np.asarray(p["left_edge"], float),
            right_edge=np.asarray(p["right_edge"], float), length=float(p["length_m"]),
            entry_index=int(p["entry_index"]), exit_index=int(p["exit_index"]),
            entry_s=float(p["entry_s"]), exit_s=float(p["exit_s"]),
            bypassed_length=float(p["bypassed_track_length_m"]),
        )
        # --- raster of the drivable area ---------------------------------------
        itw = d["image_to_world"]
        self.px_per_m: float = float(itw["px_per_m"])
        self.img_h: int = int(itw["image_height_px"])
        self.img_w: int = int(itw["image_width_px"])
        mask_path = data_dir / f"{name}_mask.png"
        img = _read_png_gray(mask_path)
        grid = np.zeros(img.shape, np.uint8)
        grid[img >= 200] = self.MAIN
        grid[(img >= 100) & (img < 200)] = self.PIT
        # flip rows so that row index grows with world y
        self.grid = np.ascontiguousarray(grid[::-1, :])
        self.grid_h, self.grid_w = self.grid.shape
        main = self.grid == self.MAIN
        self.dist_grid = distance_transform_edt(main) / self.px_per_m
        drivable = self.grid > 0
        self.dist_grid_any = distance_transform_edt(drivable) / self.px_per_m

    # ------------------------------------------------------------------ geometry
    def world_to_cell(self, x, y):
        """World (m) -> (row, col) in the flipped grid. Vectorised."""
        col = np.asarray(np.floor(np.asarray(x) * self.px_per_m), int)
        row = np.asarray(np.floor(np.asarray(y) * self.px_per_m), int)
        return row, col

    def cell_value(self, x, y) -> np.ndarray:
        """Grid value at world position(s); 0 outside the image."""
        row, col = self.world_to_cell(x, y)
        inside = (row >= 0) & (row < self.grid_h) & (col >= 0) & (col < self.grid_w)
        out = np.zeros(np.shape(row), np.uint8)
        r = np.clip(row, 0, self.grid_h - 1)
        c = np.clip(col, 0, self.grid_w - 1)
        out[...] = self.grid[r, c]
        out[~inside] = 0
        return out

    def on_track(self, x, y, allow_pit: bool = False) -> np.ndarray:
        v = self.cell_value(x, y)
        return (v > 0) if allow_pit else (v == self.MAIN)

    def distance_to_edge(self, x, y, allow_pit: bool = False) -> np.ndarray:
        """Distance [m] from position(s) to the nearest off-track cell (0 if outside)."""
        row, col = self.world_to_cell(x, y)
        r = np.clip(row, 0, self.grid_h - 1)
        c = np.clip(col, 0, self.grid_w - 1)
        g = self.dist_grid_any if allow_pit else self.dist_grid
        return g[r, c]

    def width_at(self, s: float) -> float:
        return float(self.width[int(round(s / self.spacing)) % self.n])

    # --------------------------------------------------------------------- lidar
    def lidar(self, x: float, y: float, yaw: float, angles: np.ndarray, max_range: float,
              step: float = 0.2, allow_pit: bool = False) -> np.ndarray:
        """Distances [m] along rays at ``yaw + angles`` until the first off-track cell.

        Rays are marched on the raster with ``step`` metres resolution (vectorised).
        """
        n_steps = int(np.ceil(max_range / step))
        dists = (np.arange(1, n_steps + 1) * step)[None, :]  # (1, S)
        a = (yaw + angles)[:, None]  # (R, 1)
        px = x + np.cos(a) * dists
        py = y + np.sin(a) * dists
        v = self.cell_value(px, py)
        hit = ~((v > 0) if allow_pit else (v == self.MAIN))  # (R, S)
        any_hit = hit.any(axis=1)
        first = np.where(any_hit, hit.argmax(axis=1), n_steps - 1)
        out = (first + 1) * step
        out[~any_hit] = max_range
        return np.minimum(out, max_range)

    # ---------------------------------------------------------------- utilities
    def start_pose(self, s: float = 0.0, lateral: float = 0.0):
        """World pose (x, y, yaw) at arc length s with a lateral offset (+ left)."""
        p = self.point_at(s)
        h = self.heading_at(s)
        n = np.array([-np.sin(h), np.cos(h)])
        q = p + lateral * n
        return float(q[0]), float(q[1]), h

    def polygon(self):
        """Closed polygon of the main track (left edge forward, right edge back)."""
        return np.vstack([self.left_edge, self.right_edge[::-1]])


def _read_png_gray(path: Path) -> np.ndarray:
    """Read an 8-bit PNG as a 2-D uint8 array without depending on OpenCV."""
    try:
        import imageio.v3 as iio

        img = np.asarray(iio.imread(path))
    except ImportError:  # pragma: no cover
        import matplotlib.image as mpimg

        img = mpimg.imread(path)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.uint8)


def available_tracks() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.json"))
