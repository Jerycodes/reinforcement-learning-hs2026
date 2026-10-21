"""Simple 2-D top-down viewer (matplotlib) for live watching and video frames.

Used by ``KartLapEnv.render()``: ``render_mode="human"`` opens an interactive
window, ``render_mode="rgb_array"`` returns frames (e.g. for GIF/MP4 export).
"""
from __future__ import annotations

import numpy as np


class Viewer2D:
    def __init__(self, track, interactive: bool = True, follow: bool = False, window: float = 45.0,
                 figsize=(10, 8), dpi: int = 90):
        import matplotlib

        if not interactive:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Polygon

        self.plt = plt
        self.track = track
        self.interactive = interactive
        self.follow = follow
        self.window = window
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor("#1d1f24")
        self.fig.patch.set_facecolor("#1d1f24")
        self.ax.tick_params(colors="#888")
        for sp in self.ax.spines.values():
            sp.set_color("#444")

        # static track drawing
        pit = np.vstack([track.pit.left_edge, track.pit.right_edge[::-1]])
        self.ax.add_patch(Polygon(pit, closed=True, fc="#4a4d55", ec="none", zorder=1))
        self.ax.add_patch(Polygon(track.polygon(), closed=True, fc="#5c6069", ec="#e6e6e6", lw=1.2, zorder=2))
        self.ax.plot(*np.vstack([track.centerline, track.centerline[:1]]).T, color="#8a8f99", lw=0.6,
                     ls="--", zorder=3)
        sf = np.vstack([track.left_edge[0], track.right_edge[0]])
        self.ax.plot(sf[:, 0], sf[:, 1], color="white", lw=3, zorder=4)
        for k in range(0, track.n, 100):
            self.ax.annotate(f"{k}", track.centerline[k], color="#aaa", fontsize=7, zorder=4)
        x0, x1, y0, y1 = track.bounds
        self.ax.set_xlim(x0 - 3, x1 + 3)
        self.ax.set_ylim(y0 - 3, y1 + 3)

        # dynamic artists
        self.kart_patch = Polygon(np.zeros((4, 2)), closed=True, fc="#ff5a36", ec="white", lw=1.0, zorder=10)
        self.ax.add_patch(self.kart_patch)
        self.heading_line, = self.ax.plot([], [], color="white", lw=1.5, zorder=11)
        self.lidar = LineCollection([], colors="#3fd0ff", linewidths=0.6, alpha=0.55, zorder=6)
        self.ax.add_collection(self.lidar)
        self.trail, = self.ax.plot([], [], color="#ffb000", lw=1.0, alpha=0.8, zorder=5)
        self.trail_xy: list[tuple[float, float]] = []
        self.hud = self.ax.text(0.01, 0.99, "", transform=self.ax.transAxes, va="top", ha="left",
                                color="white", fontsize=10, family="monospace", zorder=20,
                                bbox=dict(boxstyle="round", fc="#000000aa", ec="none"))
        self.title = self.ax.set_title(track.name, color="white")
        if interactive:
            plt.ion()
            self.fig.show()

    def draw(self, env):
        k = env.kart
        self.kart_patch.set_xy(k.footprint())
        hx = k.x + np.cos(k.yaw) * k.p.length * 0.8
        hy = k.y + np.sin(k.yaw) * k.p.length * 0.8
        self.heading_line.set_data([k.x, hx], [k.y, hy])
        ang = k.yaw + env.lidar_angles
        ends = np.stack([k.x + np.cos(ang) * env.lidar_dist, k.y + np.sin(ang) * env.lidar_dist], axis=1)
        self.lidar.set_segments([[(k.x, k.y), tuple(e)] for e in ends])
        if env.step_count <= 1:
            self.trail_xy = []
        self.trail_xy.append((k.x, k.y))
        if len(self.trail_xy) > 600:
            self.trail_xy = self.trail_xy[-600:]
        tr = np.array(self.trail_xy)
        self.trail.set_data(tr[:, 0], tr[:, 1])
        last = f"{env.lap_times[-1]:6.2f}s" if env.lap_times else "   -  "
        self.hud.set_text(
            f"t {env.t:6.1f}s  lap {env.laps_done + 1}  lap time {env.t - env.lap_start_t:6.2f}s  last {last}\n"
            f"v {k.speed * 3.6:5.1f} km/h  s {env.s:5.0f} m  steer {np.degrees(k.steer):+5.1f} deg  "
            f"thr {env.prev_action[1]:+.2f}" + ("  OFF TRACK" if env.off_track else "")
        )
        if self.follow:
            self.ax.set_xlim(k.x - self.window / 2, k.x + self.window / 2)
            self.ax.set_ylim(k.y - self.window / 2, k.y + self.window / 2)
        if self.interactive:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)
            return None
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        self.plt.close(self.fig)
