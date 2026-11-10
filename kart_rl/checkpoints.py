"""Load PPO weights with explicitly rebuilt, portable training schedules."""

import json
from zipfile import ZipFile


def load_portable_ppo(path, ppo_config, *, learning_rate=None, **kwargs):
    """Replace cloudpickled schedules before SB3 deserializes them.

    Python 3.14 schedule bytecode from the original Mac checkpoint cannot be
    executed by Python 3.12 on Windows. Numerical policy/optimizer weights are
    still loaded by SB3. This deliberately uses a constant learning rate from
    the explicit config, rather than resuming the old linear schedule.
    """
    from stable_baselines3 import PPO

    with ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
    clip_vf = data.get("clip_range_vf")
    if isinstance(clip_vf, dict):
        raise TypeError("scheduled clip_range_vf needs an explicit portable replacement")
    lr = float(learning_rate if learning_rate is not None else ppo_config["learning_rate"])
    return PPO.load(
        path,
        custom_objects={
            "learning_rate": lr,
            "lr_schedule": None,
            "clip_range": float(ppo_config["clip_range"]),
            "clip_range_vf": clip_vf,
        },
        **kwargs,
    )
