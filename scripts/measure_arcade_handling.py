"""Compare legal lateral kart-centre travel and a reproducible 100 ms steering tap."""

import math

import numpy as np

from kart_rl.game.race import ArcadeKart, RaceRoom
from kart_rl.track import Track


def legal_widths(track):
    offsets = np.arange(-8, 8.001, 0.02)
    kart = ArcadeKart()
    corners = kart.footprint()
    body = np.vstack([corners, (corners + np.roll(corners, 1, axis=0)) / 2])
    widths = []
    for center, forward, left in zip(track.points, track.tangent, track.normal):
        probe = center + offsets[:, None, None] * left + body[:, 0, None] * forward + body[:, 1, None] * left
        clear = track.distance_to_edge(probe[:, :, 0], probe[:, :, 1]).min(axis=1) >= 0.1
        mid = len(offsets) // 2
        assert clear[mid]
        lo = np.flatnonzero(~clear[:mid])[-1] + 1
        hi = mid + np.flatnonzero(~clear[mid:])[0] - 1
        widths.append(offsets[hi] - offsets[lo])
    return np.array(widths)


def tap(speed, previous):
    player = RaceRoom(previous=previous).join()
    driver, kart = player.keyboard, player.racer.env.kart
    kart.reset(0, 0, 0, speed)
    for tick in range(15):
        driver.set_keys({"left": tick < 3})
        command = driver.action(1 / 30)
        kart.step(command[0], 0.05, 1 / 30)
    return math.degrees(kart.yaw)


if __name__ == "__main__":
    for previous, name in [(True, "race_inn_roggwil"), (False, "roggwil_arcade")]:
        track = Track(name)
        print(name, "length", track.length, "width min/median", track.width.min(), np.median(track.width))
        free = legal_widths(track)
        print("legal centre span min/median", free.min(), np.median(free))
        print("tap 10/22 m/s", *[tap(v, previous) for v in (10, 22)])
