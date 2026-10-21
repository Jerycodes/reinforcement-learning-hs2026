#!/usr/bin/env python3
"""Extract a drivable track description from a top-down track map image.

The pipeline was written for the Race-Inn Roggwil map (grey asphalt, black
borders, white pit-lane markings) but every tunable is a CLI flag so it can be
re-run on other maps.

Pipeline
--------
1. threshold the asphalt grey level -> binary track mask
2. bridge markings that cut the track (start/finish line), fill small holes
   (arrows, text), light morphological closing
3. skeletonise the mask and prune all spurs (iteratively delete end pixels;
   the loop and the pit lane survive because neither has an end point)
4. build the skeleton graph: 2 junctions, 3 paths. The shortest junction-to-
   junction path is the pit lane, the other two form the main loop
5. smooth the loop, orient it in the driving direction, start it at the
   start/finish line, scale it so the loop length equals the known length
6. resample at a fixed spacing, measure left/right half widths by casting rays
   in the mask, re-centre, smooth again
7. compute heading, curvature, edge polylines (main track and pit lane)
8. write JSON + mask PNG + control plots

Usage (defaults are the Race-Inn Roggwil values)::

    python tools/extract_track.py --image assets/track_map.png \
        --length 640 --out kart_rl/data/race_inn_roggwil.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize

# --------------------------------------------------------------------------- #
# 1. mask
# --------------------------------------------------------------------------- #


def build_mask(gray: np.ndarray, lo: int, hi: int, bridges: list[tuple[int, int, int, int]],
               hole_area: int, object_area: int, close_px: int) -> np.ndarray:
    """Binary mask of the asphalt."""
    mask = ((gray >= lo) & (gray <= hi)).astype(np.uint8)
    # bridge markings that cut the track completely (e.g. start/finish line)
    for x0, x1, y0, y1 in bridges:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * (x1 - x0) + 1, 3))
        roi = mask[y0:y1, x0 - (x1 - x0): x1 + (x1 - x0)]
        closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k)
        mask[y0:y1, x0 - (x1 - x0): x1 + (x1 - x0)] = closed
    mask = remove_small_holes(mask.astype(bool), max_size=hole_area)
    mask = remove_small_objects(mask, max_size=object_area)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 2:  # keep the largest component only
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    return mask.astype(bool)


# --------------------------------------------------------------------------- #
# 2. skeleton graph
# --------------------------------------------------------------------------- #

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _degree_map(skel: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    return cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) * skel


def prune_spurs(skel: np.ndarray, max_iter: int = 10000) -> np.ndarray:
    """Delete end pixels until none are left (keeps cycles only)."""
    skel = skel.copy()
    for _ in range(max_iter):
        deg = _degree_map(skel)
        ends = skel & (deg <= 1)
        if not ends.any():
            break
        skel[ends] = False
    return skel


def skeleton_graph(skel: np.ndarray):
    """Return (junction_nodes, paths).

    junction_nodes: list of arrays of (row, col) pixels per junction cluster
    paths: list of dicts {'a': node_id, 'b': node_id, 'pixels': (N,2) array}
    """
    deg = _degree_map(skel)
    junction = skel & (deg >= 3)
    n, lab = cv2.connectedComponents(junction.astype(np.uint8), connectivity=8)
    node_of = {}
    nodes = []
    for i in range(1, n):
        pix = np.argwhere(lab == i)
        nodes.append(pix)
        for r, c in pix:
            node_of[(r, c)] = i - 1
    visited = np.zeros_like(skel, bool)
    paths = []
    H, W = skel.shape
    for nid, pix in enumerate(nodes):
        for r0, c0 in pix:
            for dr, dc in _NB:
                r, c = r0 + dr, c0 + dc
                if not (0 <= r < H and 0 <= c < W) or not skel[r, c] or (r, c) in node_of or visited[r, c]:
                    continue
                # trace this path
                path = [(r0, c0), (r, c)]
                visited[r, c] = True
                pr, pc = r0, c0
                cr, cc = r, c
                end_node = None
                while True:
                    nxt = None
                    for dr2, dc2 in _NB:
                        nr, nc = cr + dr2, cc + dc2
                        if not (0 <= nr < H and 0 <= nc < W) or not skel[nr, nc]:
                            continue
                        if (nr, nc) == (pr, pc):
                            continue
                        if (nr, nc) in node_of:
                            if node_of[(nr, nc)] == nid and len(path) < 3:
                                continue  # still inside the start cluster
                            end_node = node_of[(nr, nc)]
                            path.append((nr, nc))
                            nxt = None
                            break
                        if not visited[nr, nc]:
                            nxt = (nr, nc)
                            break
                    if end_node is not None or nxt is None:
                        break
                    visited[nxt] = True
                    path.append(nxt)
                    pr, pc, cr, cc = cr, cc, nxt[0], nxt[1]
                if end_node is None:
                    continue
                paths.append({"a": nid, "b": end_node, "pixels": np.array(path, float)})
    return simplify_graph(nodes, paths)


def simplify_graph(nodes, paths):
    """Merge pass-through nodes (exactly two incident paths) into one path."""
    changed = True
    while changed:
        changed = False
        for nid in range(len(nodes)):
            inc = [p for p in paths if nid in (p["a"], p["b"])]
            if len(inc) != 2 or inc[0] is inc[1]:
                continue
            p, q = inc
            # orient both so that they meet at nid: p ends at nid, q starts at nid
            pp = p["pixels"] if p["b"] == nid else p["pixels"][::-1]
            qq = q["pixels"] if q["a"] == nid else q["pixels"][::-1]
            a = p["a"] if p["b"] == nid else p["b"]
            b = q["b"] if q["a"] == nid else q["a"]
            paths = [r for r in paths if r is not p and r is not q]
            paths.append({"a": a, "b": b, "pixels": np.vstack([pp, qq[1:]])})
            changed = True
            break
    used = sorted({p["a"] for p in paths} | {p["b"] for p in paths})
    remap = {old: new for new, old in enumerate(used)}
    for p in paths:
        p["a"], p["b"] = remap[p["a"]], remap[p["b"]]
    return [nodes[i] for i in used], paths


# --------------------------------------------------------------------------- #
# 3. geometry helpers (all in (x, y) order from here on)
# --------------------------------------------------------------------------- #


def polyline_length(p: np.ndarray, closed: bool) -> float:
    q = np.vstack([p, p[:1]]) if closed else p
    return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


def smooth(p: np.ndarray, sigma: float, closed: bool) -> np.ndarray:
    mode = "wrap" if closed else "nearest"
    return np.stack([gaussian_filter1d(p[:, i], sigma, mode=mode) for i in range(2)], axis=1)


def resample(p: np.ndarray, spacing: float, closed: bool) -> np.ndarray:
    q = np.vstack([p, p[:1]]) if closed else p
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = int(round(total / spacing))
    if closed:
        s_new = np.linspace(0.0, total, n, endpoint=False)
    else:
        s_new = np.linspace(0.0, total, n + 1)
    return np.stack([np.interp(s_new, s, q[:, i]) for i in range(2)], axis=1)


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def heading_and_curvature(p: np.ndarray, closed: bool):
    """Heading [rad] and signed curvature [1/m] (+ = left turn) per sample."""
    if closed:
        d = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / 2.0
    else:
        d = np.gradient(p, axis=0)
    heading = np.arctan2(d[:, 1], d[:, 0])
    ds = np.linalg.norm(d, axis=1)
    if closed:
        dh = wrap_angle(np.roll(heading, -1) - np.roll(heading, 1)) / 2.0
    else:
        dh = np.gradient(np.unwrap(heading))
    return heading, dh / ds


def fill_runs(values: np.ndarray, bad: np.ndarray, closed: bool) -> np.ndarray:
    """Linearly interpolate over runs of `bad` samples (circular if closed)."""
    v = values.astype(float).copy()
    n = len(v)
    if not bad.any() or bad.all():
        return v
    idx = np.arange(n)
    good = ~bad
    if closed:
        # unroll so that the array starts at a good sample
        start = int(np.argmax(good))
        order = np.roll(idx, -start)
        xg = idx[good[order]]
        vg = v[order][good[order]]
        # append the first good sample at the end for wrap-around
        xg = np.r_[xg, n]
        vg = np.r_[vg, vg[0]]
        filled = np.interp(idx, xg, vg)
        out = np.empty(n)
        out[order] = filled
        return out
    return np.interp(idx, idx[good], v[good])


def cast_width(mask: np.ndarray, p_px: np.ndarray, heading: np.ndarray, max_px: float, step: float = 0.25):
    """Distance from each point to the mask border on the left and right (pixels)."""
    H, W = mask.shape
    nx, ny = -np.sin(heading), np.cos(heading)  # left normal in image coords (y down -> "left" is flipped later)
    out = np.zeros((len(p_px), 2))
    for i, (x, y) in enumerate(p_px):
        for side, sign in ((0, 1.0), (1, -1.0)):
            d = 0.0
            while d < max_px:
                xx, yy = x + sign * nx[i] * d, y + sign * ny[i] * d
                xi, yi = int(round(xx)), int(round(yy))
                if not (0 <= xi < W and 0 <= yi < H) or not mask[yi, xi]:
                    break
                d += step
            out[i, side] = d
    return out


# --------------------------------------------------------------------------- #
# 4. main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="assets/track_map.png")
    ap.add_argument("--out", default="kart_rl/data/race_inn_roggwil.json")
    ap.add_argument("--mask-out", default="kart_rl/data/race_inn_roggwil_mask.png")
    ap.add_argument("--plot", default="docs/img/track_extraction.png")
    ap.add_argument("--name", default="Race-Inn Roggwil")
    ap.add_argument("--length", type=float, default=640.0, help="known centreline length [m]")
    ap.add_argument("--grey", type=int, nargs=2, default=(135, 172), help="asphalt grey range")
    ap.add_argument("--bridge", type=int, nargs=4, action="append", default=[[453, 477, 1040, 1160]],
                    metavar=("X0", "X1", "Y0", "Y1"), help="box with a marking that cuts the track")
    ap.add_argument("--hole-area", type=int, default=4000)
    ap.add_argument("--object-area", type=int, default=5000)
    ap.add_argument("--close-px", type=int, default=7)
    ap.add_argument("--start-finish", type=float, nargs=2, default=(465, 1100), metavar=("X", "Y"),
                    help="pixel on the start/finish line")
    ap.add_argument("--flow", type=float, nargs=4, default=(600, 1100, -1, 0), metavar=("X", "Y", "DX", "DY"),
                    help="pixel on the track + driving direction there (image coords)")
    ap.add_argument("--pit-hint", type=float, nargs=2, default=(1390, 900), metavar=("X", "Y"),
                    help="pixel that lies on the pit lane")
    ap.add_argument("--spacing", type=float, default=1.0, help="sample spacing [m]")
    ap.add_argument("--smooth-m", type=float, default=2.0, help="gaussian smoothing sigma [m]")
    ap.add_argument("--recentre-iters", type=int, default=3)
    ap.add_argument("--bridge-m", type=float, default=6.0,
                    help="radius around each pit junction that is cut out of the loop and bridged [m]")
    ap.add_argument("--pit-speed-limit", type=float, default=30.0, help="documentation only [km/h]")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    mask = build_mask(gray, args.grey[0], args.grey[1], [tuple(b) for b in args.bridge],
                      args.hole_area, args.object_area, args.close_px)
    skel = prune_spurs(skeletonize(mask))
    nodes, paths = skeleton_graph(skel)
    if len(nodes) != 2 or len(paths) != 3:
        raise SystemExit(f"expected 2 junctions / 3 paths, got {len(nodes)} / {len(paths)}; "
                         "tune --grey/--close-px/--hole-area")
    lengths = [polyline_length(p["pixels"], closed=False) for p in paths]
    # the pit lane is the junction-to-junction path that passes closest to --pit-hint
    hint = np.array(args.pit_hint)[::-1]  # (x, y) -> (row, col)
    hint_dist = [float(np.linalg.norm(p["pixels"] - hint, axis=1).min()) for p in paths]
    pit_i = int(np.argmin(hint_dist))
    main_paths = [p for i, p in enumerate(paths) if i != pit_i]
    pit = paths[pit_i]
    print(f"paths (px): {[round(l) for l in lengths]}, distance to pit hint {[round(d) for d in hint_dist]} "
          f"-> pit lane = path {pit_i}")

    # --- main loop pixel sequence (row,col -> x,y) ----------------------------
    a, b = main_paths
    pa = a["pixels"][:, ::-1]
    pb = b["pixels"][:, ::-1]
    # orient b so that it continues from the end of a
    if np.linalg.norm(pb[0] - pa[-1]) > np.linalg.norm(pb[-1] - pa[-1]):
        pb = pb[::-1]
    loop_px = np.vstack([pa, pb[1:-1]])  # closed loop (last connects to first)
    # smooth in pixel units to get rid of the staircase, then measure length
    loop_px = smooth(loop_px, sigma=4.0, closed=True)
    px_per_m0 = polyline_length(loop_px, closed=True) / args.length
    # the skeleton detours towards the junction point at each pit mouth:
    # cut those parts out and bridge them with a straight line (re-centred later)
    junction_xy = [node[:, ::-1].mean(axis=0) for node in nodes]
    keep = np.ones(len(loop_px), bool)
    for jx in junction_xy:
        keep &= np.linalg.norm(loop_px - jx, axis=1) > args.bridge_m * px_per_m0
    loop_px = resample(loop_px[keep], 1.0, closed=True)

    # --- driving direction -----------------------------------------------------
    fx, fy, fdx, fdy = args.flow
    i_flow = int(np.argmin(np.linalg.norm(loop_px - [fx, fy], axis=1)))
    tangent = loop_px[(i_flow + 3) % len(loop_px)] - loop_px[(i_flow - 3) % len(loop_px)]
    if tangent[0] * fdx + tangent[1] * fdy < 0:
        loop_px = loop_px[::-1]
    # start at start/finish
    i_sf = int(np.argmin(np.linalg.norm(loop_px - list(args.start_finish), axis=1)))
    loop_px = np.roll(loop_px, -i_sf, axis=0)

    # --- scale --------------------------------------------------------------
    px_per_m = polyline_length(loop_px, closed=True) / args.length
    print(f"scale: {px_per_m:.3f} px/m (image {W}x{H} -> {W/px_per_m:.1f} x {H/px_per_m:.1f} m)")

    def px_to_world(p):
        return np.stack([p[:, 0] / px_per_m, (H - p[:, 1]) / px_per_m], axis=1)

    def world_to_px(p):
        return np.stack([p[:, 0] * px_per_m, H - p[:, 1] * px_per_m], axis=1)

    # --- split the mask into main track / pit lane (Voronoi by skeleton path) --
    pit_px = smooth(pit["pixels"][:, ::-1], sigma=4.0, closed=False)
    main_only = np.zeros((H, W), np.uint8)
    pit_only = np.zeros((H, W), np.uint8)
    cv2.polylines(main_only, [loop_px.astype(np.int32)], True, 1, 1)
    # the pit path starts/ends at the junction point inside the main track: trim
    # its ends for the split so that the main track keeps its full width there
    pit_trim = resample(pit_px, 1.0, closed=False)
    t = int(args.bridge_m * px_per_m0)
    cv2.polylines(pit_only, [pit_trim[t:-t].astype(np.int32)], False, 1, 1)
    d_main = cv2.distanceTransform(1 - main_only, cv2.DIST_L2, 5)
    d_pit = cv2.distanceTransform(1 - pit_only, cv2.DIST_L2, 5)
    pit_mask = mask & (d_pit < d_main)
    main_mask = mask & ~(d_pit < d_main)

    # --- resample, measure width, re-centre, smooth --------------------------
    # cast_width works in image space: image heading = -world heading, and the
    # image-space +normal is the driver's RIGHT (y axis flipped) -> dist[:,0]
    # is the right half width, dist[:,1] the left half width.
    # one real smoothing pass (removes skeleton noise), then re-centre with only
    # light smoothing in between: repeated strong smoothing would shrink the
    # hairpins and pull the centreline off the track.
    loop = resample(px_to_world(loop_px), args.spacing, closed=True)
    loop = smooth(loop, args.smooth_m / args.spacing, closed=True)
    for it in range(args.recentre_iters):
        loop = resample(loop, args.spacing, closed=True)
        heading, _ = heading_and_curvature(loop, closed=True)
        dist = cast_width(main_mask, world_to_px(loop), -heading, max_px=12 * px_per_m)
        right_px, left_px = dist[:, 0], dist[:, 1]
        shift = (right_px - left_px) / 2.0  # >0: centre is too far left -> move right
        # at the pit mouths one side is open: do not re-centre into the pit lane
        cap = 1.25 * np.median(np.r_[left_px, right_px])
        shift[(left_px > cap) | (right_px > cap)] = 0.0
        nrm = np.stack([-np.sin(heading), np.cos(heading)], axis=1)  # left normal, world
        loop = loop - nrm * (shift / px_per_m)[:, None]
        loop = smooth(loop, 0.5 / args.spacing, closed=True)
        if it == 0:
            # refine the main/pit split with the re-centred loop
            main_only[:] = 0
            cv2.polylines(main_only, [world_to_px(loop).astype(np.int32)], True, 1, 1)
            d_main = cv2.distanceTransform(1 - main_only, cv2.DIST_L2, 5)
            pit_mask = mask & (d_pit < d_main)
            main_mask = mask & ~(d_pit < d_main)
    loop = smooth(loop, 1.0 / args.spacing, closed=True)
    # final scale correction so the loop is exactly args.length long
    loop *= args.length / polyline_length(loop, closed=True)
    loop = resample(loop, args.spacing, closed=True)
    heading, curvature = heading_and_curvature(loop, closed=True)
    s_loop = np.arange(len(loop)) * args.spacing
    dist = cast_width(main_mask, world_to_px(loop), -heading, max_px=12 * px_per_m)
    right_m, left_m = dist[:, 0] / px_per_m, dist[:, 1] / px_per_m
    n_outside = int(((right_m == 0) & (left_m == 0)).sum())
    print(f"centreline samples outside the track mask: {n_outside}")
    # at the pit mouths one side is open: interpolate the edge across the mouth
    half_med = np.median(np.r_[left_m, right_m])
    bad_l = (left_m > 1.5 * half_med) | (left_m < 0.5 * half_med)
    bad_r = (right_m > 1.5 * half_med) | (right_m < 0.5 * half_med)
    n_capped = int((bad_l | bad_r).sum())
    left_m, right_m = fill_runs(left_m, bad_l, True), fill_runs(right_m, bad_r, True)
    width = left_m + right_m

    # --- pit lane -------------------------------------------------------------
    pit_w = resample(px_to_world(pit_px), args.spacing, closed=False)
    pit_w = smooth(pit_w, args.smooth_m / args.spacing, closed=False)
    # orient the pit lane in driving direction: entry = junction met first along the loop
    i_e = int(np.argmin(np.linalg.norm(loop - pit_w[0], axis=1)))
    i_x = int(np.argmin(np.linalg.norm(loop - pit_w[-1], axis=1)))
    if i_e > i_x:
        pit_w = pit_w[::-1]
        i_e, i_x = i_x, i_e
    # blend the pit lane ends onto the main centreline so the paths connect exactly
    n_blend = int(round(args.bridge_m / args.spacing))
    for end, target in ((0, loop[i_e]), (-1, loop[i_x])):
        delta = target - pit_w[end]
        w = np.linspace(1.0, 0.0, n_blend + 1)
        if end == 0:
            pit_w[: n_blend + 1] += np.outer(w, delta)
        else:
            pit_w[-(n_blend + 1):] += np.outer(w[::-1], delta)
    pit_w = resample(pit_w, args.spacing, closed=False)
    pit_heading, pit_curv = heading_and_curvature(pit_w, closed=False)
    pd = cast_width(pit_mask, world_to_px(pit_w), -pit_heading, max_px=12 * px_per_m)
    pit_right, pit_left = pd[:, 0] / px_per_m, pd[:, 1] / px_per_m
    pit_med = np.median(np.r_[pit_left, pit_right])
    pit_left = fill_runs(pit_left, (pit_left > 1.5 * pit_med) | (pit_left < 0.5 * pit_med), False)
    pit_right = fill_runs(pit_right, (pit_right > 1.5 * pit_med) | (pit_right < 0.5 * pit_med), False)

    nrm = np.stack([-np.sin(heading), np.cos(heading)], axis=1)
    left_edge = loop + nrm * left_m[:, None]
    right_edge = loop - nrm * right_m[:, None]
    pn = np.stack([-np.sin(pit_heading), np.cos(pit_heading)], axis=1)
    pit_left_edge = pit_w + pn * pit_left[:, None]
    pit_right_edge = pit_w - pn * pit_right[:, None]

    pit_len = polyline_length(pit_w, closed=False)
    bypass_len = (i_x - i_e) * args.spacing
    signed_area = 0.5 * float(np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1]))
    print(f"driving direction: {'clockwise' if signed_area < 0 else 'counter-clockwise'}")
    print(f"length {polyline_length(loop, closed=True):.1f} m, {len(loop)} samples")
    print(f"width: mean {width.mean():.2f} m, min {width.min():.2f}, max {width.max():.2f} "
          f"({n_capped} samples capped at pit mouths)")
    print(f"pit lane: {pit_len:.1f} m, entry s={i_e*args.spacing:.0f} m, exit s={i_x*args.spacing:.0f} m, "
          f"bypasses {bypass_len:.0f} m of track")
    print(f"curvature: max left {curvature.max():.3f} 1/m (R={1/curvature.max():.1f} m), "
          f"max right {curvature.min():.3f} 1/m (R={-1/curvature.min():.1f} m)")

    # --- outputs --------------------------------------------------------------
    out = {
        "name": args.name,
        "source_image": str(Path(args.image).as_posix()),
        "mask_image": str(Path(args.mask_out).as_posix()),
        "length_m": args.length,
        "sample_spacing_m": args.spacing,
        "direction": "clockwise" if signed_area < 0 else "counter-clockwise",
        "frame": "world frame in metres, x east, y north (image y axis flipped); index 0 = start/finish",
        "image_to_world": {"px_per_m": px_per_m, "image_height_px": H, "image_width_px": W,
                           "formula": "x_m = x_px / px_per_m ; y_m = (image_height_px - y_px) / px_per_m"},
        "centerline": loop.round(3).tolist(),
        "s": s_loop.round(3).tolist(),
        "heading": heading.round(5).tolist(),
        "curvature": curvature.round(5).tolist(),
        "width": width.round(3).tolist(),
        "left_edge": left_edge.round(3).tolist(),
        "right_edge": right_edge.round(3).tolist(),
        "start_finish": {"index": 0, "s": 0.0, "image_px": list(args.start_finish)},
        "pit_lane": {
            "centerline": pit_w.round(3).tolist(),
            "heading": pit_heading.round(5).tolist(),
            "width": (pit_left + pit_right).round(3).tolist(),
            "left_edge": pit_left_edge.round(3).tolist(),
            "right_edge": pit_right_edge.round(3).tolist(),
            "length_m": round(pit_len, 2),
            "entry_index": i_e, "exit_index": i_x,
            "entry_s": round(i_e * args.spacing, 2), "exit_s": round(i_x * args.spacing, 2),
            "bypassed_track_length_m": round(bypass_len, 2),
            "speed_limit_kmh": args.pit_speed_limit,
        },
        "bounds": {"x_min": float(min(left_edge[:, 0].min(), right_edge[:, 0].min(), pit_left_edge[:, 0].min())),
                   "x_max": float(max(left_edge[:, 0].max(), right_edge[:, 0].max(), pit_right_edge[:, 0].max())),
                   "y_min": float(min(left_edge[:, 1].min(), right_edge[:, 1].min(), pit_left_edge[:, 1].min())),
                   "y_max": float(max(left_edge[:, 1].max(), right_edge[:, 1].max(), pit_left_edge[:, 1].max()))},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    # mask PNG: 0 = off track, 128 = pit lane, 255 = main track
    mask_png = np.zeros((H, W), np.uint8)
    mask_png[pit_mask] = 128
    mask_png[main_mask] = 255
    Path(args.mask_out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.mask_out, mask_png)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1024:.0f} kB) and {args.mask_out}")

    # --- control plots -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[5, 1.2, 1.2])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    lp = world_to_px(loop)
    le, re = world_to_px(left_edge), world_to_px(right_edge)
    ax.plot(np.r_[le[:, 0], le[0, 0]], np.r_[le[:, 1], le[0, 1]], color="yellow", lw=1)
    ax.plot(np.r_[re[:, 0], re[0, 0]], np.r_[re[:, 1], re[0, 1]], color="yellow", lw=1)
    ax.plot(np.r_[lp[:, 0], lp[0, 0]], np.r_[lp[:, 1], lp[0, 1]], color="red", lw=1.5, label="Mittellinie")
    pp = world_to_px(pit_w)
    ax.plot(pp[:, 0], pp[:, 1], color="deepskyblue", lw=1.5, label="Boxengasse")
    for k in range(0, len(lp), 50):
        ax.annotate(f"{k}", lp[k], color="white", fontsize=7, ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="red", ec="none"))
    ax.scatter(*lp[0], s=80, marker="s", color="lime", zorder=5, label="Start/Ziel")
    for k in range(10, len(lp), 40):
        d = lp[(k + 2) % len(lp)] - lp[k - 2]
        ax.annotate("", xy=lp[k] + d, xytext=lp[k], arrowprops=dict(arrowstyle="->", color="white", lw=1.2))
    ax.set_title("Overlay auf Originalbild (Zahlen = Streckenmeter)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_axis_off()

    ax2 = fig.add_subplot(gs[0, 1])
    poly = np.vstack([left_edge, right_edge[::-1]])
    ax2.fill(poly[:, 0], poly[:, 1], color="0.6", label="Strecke")
    ppoly = np.vstack([pit_left_edge, pit_right_edge[::-1]])
    ax2.fill(ppoly[:, 0], ppoly[:, 1], color="0.8", label="Boxengasse")
    ax2.plot(np.r_[loop[:, 0], loop[0, 0]], np.r_[loop[:, 1], loop[0, 1]], "r-", lw=1)
    sc = ax2.scatter(loop[:, 0], loop[:, 1], c=curvature, cmap="coolwarm", vmin=-0.3, vmax=0.3, s=6, zorder=3)
    plt.colorbar(sc, ax=ax2, label="Krümmung [1/m] (+ = Linkskurve)")
    ax2.scatter(*loop[0], s=80, marker="s", color="lime", zorder=5)
    ax2.scatter(*loop[i_e], s=60, marker="v", color="blue", zorder=5, label="Box-Einfahrt")
    ax2.scatter(*loop[i_x], s=60, marker="^", color="blue", zorder=5, label="Box-Ausfahrt")
    ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title(f"Weltkoordinaten – {polyline_length(loop, True):.0f} m, Breite Ø {width.mean():.1f} m")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(s_loop, curvature, lw=1)
    ax3.axhline(0, color="k", lw=0.5)
    ax3.set_ylabel("Krümmung [1/m]")
    ax3.set_xlim(0, args.length)
    ax3.grid(alpha=0.3)
    ax4 = fig.add_subplot(gs[2, :], sharex=ax3)
    ax4.plot(s_loop, width, lw=1, label="Breite")
    ax4.plot(s_loop, left_m, lw=0.8, ls="--", label="links")
    ax4.plot(s_loop, right_m, lw=0.8, ls="--", label="rechts")
    ax4.axvspan(i_e * args.spacing, i_x * args.spacing, color="deepskyblue", alpha=0.15, label="Boxengasse parallel")
    ax4.set_ylabel("Breite [m]")
    ax4.set_xlabel("Streckenmeter s [m]")
    ax4.legend(loc="upper right", fontsize=8, ncol=4)
    ax4.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=110)
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
