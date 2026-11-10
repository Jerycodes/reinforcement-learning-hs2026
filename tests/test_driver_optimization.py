from types import SimpleNamespace

import pytest

from scripts.calibrate_driver import apply_calibration
from scripts.validate_optimized_driver import qualifies
from train.optimize_driver import PROFILES


def test_actor_calibration_changes_only_requested_outputs_and_does_not_accumulate():
    import torch

    head = torch.nn.Linear(3, 2)
    model = SimpleNamespace(policy=SimpleNamespace(action_net=head))
    weight, bias = head.weight.detach().clone(), head.bias.detach().clone()
    observation = torch.tensor([[1.0, 2.0, 3.0]])
    original = head(observation).detach().clone()
    apply_calibration(model, weight, bias, 0.96, 0.03, throttle_gain=1.2)
    output = head(observation)
    assert output[0, 0].item() == pytest.approx(original[0, 0].item() * 0.96, abs=1e-6)
    assert output[0, 1].item() == pytest.approx(original[0, 1].item() * 1.2 + 0.03, abs=1e-6)
    apply_calibration(model, weight, bias, 1.0, 0.0)
    assert torch.equal(head.weight, weight) and torch.equal(head.bias, bias)
    assert torch.equal(head(observation), original)


@pytest.mark.parametrize("gain,bias", [(0, 0), (-1, 0), (float("nan"), 0), (1, float("inf"))])
def test_invalid_calibration_rejected(gain, bias):
    with pytest.raises(ValueError):
        apply_calibration(None, None, None, gain, bias)


def test_optimization_profiles_do_not_change_physics_or_track_rules():
    allowed = {"learning_rate", "ent_coef", "gamma", "gae_lambda", "reward_time"}
    assert set(PROFILES) == {"gentle", "long_horizon", "time_pressure"}
    assert all(set(profile) <= allowed for profile in PROFILES.values())
    assert all(0 < profile["learning_rate"] < 1e-4 for profile in PROFILES.values())


def test_fast_candidate_is_promoted_only_when_full_run_and_reliability_pass():
    original = {"long_run": {"finished": True, "mean_flying_s": 52}, "random_finished": 9}
    candidate = {"long_run": {"finished": True, "mean_flying_s": 51.8}, "random_finished": 9}
    assert qualifies(original, candidate)
    candidate["random_finished"] = 8
    assert not qualifies(original, candidate)
    candidate["random_finished"] = 10
    candidate["long_run"]["finished"] = False
    assert not qualifies(original, candidate)
    candidate["long_run"].update(finished=True, mean_flying_s=52.1)
    assert not qualifies(original, candidate)
