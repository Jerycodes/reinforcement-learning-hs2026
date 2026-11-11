"""Build the enlarged Roggwil arcade layout and raster from identical track ribbons."""

import json

import numpy as np
from PIL import Image, ImageDraw

from kart_rl.track import DATA_DIR

SCALE = 1.35
NAME = "roggwil_arcade"


def scale_fields(obj, names, factor):
    for name in names.split():
        obj[name] = (np.asarray(obj[name]) * factor).round(6).tolist()


def main():
    circuit = json.loads((DATA_DIR / "race_inn_roggwil.json").read_text())
    fields = "centerline left_edge right_edge width length_m"
    scale_fields(circuit, fields + " s sample_spacing_m", SCALE)
    scale_fields(circuit, "curvature", 1 / SCALE)
    pit = circuit["pit_lane"]
    scale_fields(pit, fields + " entry_s exit_s bypassed_track_length_m", SCALE)
    circuit["bounds"] = {k: v * SCALE for k, v in circuit["bounds"].items()}
    circuit.update(name="Roggwil Arcade", world_scale=SCALE, mask_image=f"kart_rl/data/{NAME}_mask.png")
    frame = circuit["image_to_world"]
    frame["px_per_m"] /= SCALE
    height = frame["image_height_px"]
    mask = Image.new("L", (frame["image_width_px"], height))
    draw = ImageDraw.Draw(mask)
    for ribbon, closed, value in [(pit, False, 128), (circuit, True, 255)]:

        def pixels(edge):
            return [(int(x * frame["px_per_m"]), height - 1 - int(y * frame["px_per_m"])) for x, y in edge]

        left, right = pixels(ribbon["left_edge"]), pixels(ribbon["right_edge"])
        for i in range(len(left) if closed else len(left) - 1):
            j = (i + 1) % len(left)
            draw.polygon([left[i], right[i], right[j], left[j]], fill=value)
    mask.save(DATA_DIR / f"{NAME}_mask.png")
    (DATA_DIR / f"{NAME}.json").write_text(json.dumps(circuit, separators=(",", ":")), encoding="utf8")


if __name__ == "__main__":
    main()
