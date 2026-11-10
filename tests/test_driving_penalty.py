import json
from zipfile import ZipFile

import gymnasium as gym
import pytest

from kart_rl.checkpoints import load_portable_ppo
from scripts.compare_driving_penalty import penalty_config, summarize_trials, write_report


def test_time_cost_is_per_second_and_control_is_unchanged():
    source = {"dt": 0.05, "reward_progress": 1, "reward_off_track": -20, "kart": {"mu": 1}}
    control = penalty_config(source, None)
    penalty = penalty_config(source, 25)
    assert control == source
    assert penalty["reward_time"] == pytest.approx(-0.002)
    assert penalty["reward_time"] / source["dt"] == pytest.approx(-1 / 25)
    assert {k: v for k, v in penalty.items() if k != "reward_time"} == source
    penalty["kart"]["mu"] = 0
    assert source["kart"]["mu"] == 1


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_time_scales_rejected(scale):
    with pytest.raises(ValueError):
        penalty_config({}, scale)


def test_failed_evaluation_is_not_counted_as_a_fast_driver():
    result = summarize_trials(
        [
            {"arm": "time_penalty", "finished": True, "mean_flying_s": 52},
            {"arm": "time_penalty", "finished": False, "mean_flying_s": 48},
            {"arm": "control", "finished": False, "mean_flying_s": None},
        ]
    )
    assert result[0]["runs"] == 2 and result[0]["finished"] == 1
    assert result[0]["mean_flying_s"] == 52 and result[0]["std_s"] is None
    assert result[1]["mean_flying_s"] is None


def test_portable_loader_replaces_serialized_schedules_and_preserves_weights(tmp_path):
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    env = gym.make("CartPole-v1")
    try:
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, policy_kwargs={"net_arch": [8]}, seed=0)

        def incompatible_schedule(_):
            raise RuntimeError("old Python schedule must never execute")

        model.learning_rate = incompatible_schedule
        model.lr_schedule = incompatible_schedule
        model.clip_range = incompatible_schedule
        path = tmp_path / "model.zip"
        model.save(path)
        loaded = load_portable_ppo(path, {"learning_rate": 0.0003, "clip_range": 0.2}, device="cpu")
        assert loaded.lr_schedule(1) == pytest.approx(0.0003)
        assert loaded.clip_range(1) == pytest.approx(0.2)
        assert all(
            torch.equal(value, loaded.policy.state_dict()[key])
            for key, value in model.policy.state_dict().items()
        )
    finally:
        env.close()


def test_unknown_value_clip_schedule_is_not_silently_changed(tmp_path):
    path = tmp_path / "model.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps({"clip_range_vf": {":serialized:": "unknown"}}))
    with pytest.raises(TypeError):
        load_portable_ppo(path, {"learning_rate": 0.0003, "clip_range": 0.2})


def test_report_shows_real_lap_times_and_no_pit_stops(tmp_path):
    manifest = {
        "steps": 40960,
        "seeds": [0],
        "eval_laps": 5,
        "scale": 25,
        "dt": 0.05,
        "learning_rate": 1e-4,
        "reward_off_track": -20,
    }
    original = {"episode_done": "<wall>", "standing_s": 54, "mean_flying_s": 52, "best_flying_s": 51.9}
    row = {
        "arm": "control",
        "seed": 0,
        "finished": True,
        "episode_done": "laps",
        "standing_s": 54,
        "mean_flying_s": 52,
        "best_flying_s": 51.9,
    }
    write_report(tmp_path, manifest, original, [row])
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "&lt;wall&gt;" in page and "<wall>" not in page
    assert "Kein Reifenverschleiss" in page
    assert json.loads((tmp_path / "comparison.json").read_text())["summary"][0]["finished"] == 1
