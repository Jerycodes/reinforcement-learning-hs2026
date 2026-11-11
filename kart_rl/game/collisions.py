"""Swept oriented bumper rectangles. All positions, margins and distances are metres."""

import math
from itertools import combinations

import numpy as np

SKIN = 0.002


def axes(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(((c, s), (-s, c)))


def radius(basis, axis, params):
    local = abs(basis @ axis)
    return (params.length * local[0] + params.width * local[1]) / 2


def swept_contact(a, b, a0, b0):
    """Continuous separating-axis intervals for translation, bounded over rotation.

    Return the entry-side normal (A -> B) and the relative displacement that
    must be removed along it. This also detects passing through and out again.
    """
    start = b0[:2] - a0[:2]
    end = b.state[:2] - a.state[:2]
    delta = end - start
    reach = (math.hypot(a.p.length, a.p.width) + math.hypot(b.p.length, b.p.width)) / 2
    if np.max(abs(start)) > reach + np.max(abs(delta)) + SKIN:
        return None
    ab, bb = axes(a0[2]), axes(b0[2])

    def turn_bound(kart, old):
        angle = (kart.yaw - old[2] + math.pi) % (2 * math.pi) - math.pi
        return math.hypot(kart.p.length, kart.p.width) * abs(angle) / 2

    padding = turn_bound(a, a0) + turn_bound(b, b0)
    enter, leave = -math.inf, 1.0
    nearest, initial_gap = None, -math.inf
    hit_axis = None
    for axis in (*ab, *bb):
        extent = radius(ab, axis, a.p) + radius(bb, axis, b.p) + padding + SKIN
        d, v = float(start @ axis), float(delta @ axis)
        gap = abs(d) - extent
        side = axis if d >= 0 else -axis
        if gap > initial_gap:
            nearest, initial_gap = side, gap
        if abs(v) < 1e-10:
            if gap >= 0:
                return None
            continue
        t1, t2 = sorted(((-extent - d) / v, (extent - d) / v))
        if t1 > enter:
            enter, hit_axis = t1, (-axis if v > 0 else axis)
        leave = min(leave, t2)
        if enter > leave:
            return None
    if leave < 0 or enter > 1:
        return None
    normal = nearest if enter <= 0 else hit_axis
    separation = radius(ab, normal, a.p) + radius(bb, normal, b.p) + padding + SKIN
    depth = separation - float(end @ normal)
    return (normal, depth) if depth > 1e-7 else None


def impulse(a, b, normal):
    """Equal-mass normal impulse; preserve tangential motion in world coordinates."""
    ra, rb = axes(a.yaw), axes(b.yaw)
    va, vb = ra.T @ a.state[3:5], rb.T @ b.state[3:5]
    closing = (va - vb) @ normal
    if closing > 0:
        j = normal * (0.53 * closing)
        a.state[3:5] = ra @ (va - j)
        b.state[3:5] = rb @ (vb + j)


def legal_shift(env, shift):
    old = env.kart.state[:2].copy()
    env.kart.state[:2] += shift
    if env._clearance() >= env.cfg.off_track_margin:
        return True
    env.kart.state[:2] = old
    return False


def resolve(players, previous):
    """Only publish legal end poses. A jam that doesn't converge keeps its last safe pose."""
    cars = [p.racer.env for p in players]
    pairs = list(combinations(range(len(cars)), 2))
    changed = set()
    for i, env in enumerate(cars):
        if env.off_track and env._clearance() < env.cfg.off_track_margin:
            # Slide along the barrier and face down-track. The raster can be
            # tighter than the polygon; accept only poses with full clearance.
            k = env.kart
            tangent = env.track.heading_at(env.s)
            if math.cos(k.yaw - tangent) < 0:
                tangent += math.pi
            k.state[:2] = previous[i][:2]
            k.state[2] += 0.22 * ((tangent - k.yaw + math.pi) % (2 * math.pi) - math.pi)
            inward = env.track.normal[env.idx] * -np.sign(env.lateral)
            k.state[:2] += inward * 0.12
            if env._clearance() < env.cfg.off_track_margin:
                k.state[:3] = previous[i][:3]
            k.state[3:5] = (max(0, k.vx * 0.60), 0)
            changed.add(i)
    for _ in range(10):
        contacts = False
        for i, j in pairs:
            a, b = cars[i].kart, cars[j].kart
            hit = swept_contact(a, b, previous[i], previous[j])
            if hit is None:
                continue
            contacts = True
            changed.update((i, j))
            normal, depth = hit
            push = normal * (depth + SKIN)
            moved_a = legal_shift(cars[i], -push / 2)
            moved_b = legal_shift(cars[j], push / 2)
            if moved_a != moved_b:
                if moved_a:
                    legal_shift(cars[i], -push / 2)
                else:
                    legal_shift(cars[j], push / 2)
            impulse(a, b, normal)
        if not contacts:
            break
    else:
        # Freeze only the unresolvable cluster, expanding it if resetting a
        # member would put it across another moving kart's swept path.
        frozen = set()
        for _ in range(len(cars) + 1):
            blocked = set()
            for i, j in pairs:
                if {i, j} <= frozen:
                    continue
                if swept_contact(cars[i].kart, cars[j].kart, previous[i], previous[j]):
                    blocked.update((i, j))
            if not blocked:
                break
            for i in blocked:
                cars[i].kart.state[:3] = previous[i][:3]
                cars[i].kart.state[3:5] *= 0.5
            frozen.update(blocked)
        changed.update(frozen)
    return changed
