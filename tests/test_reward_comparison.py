import json

import pytest

from scripts.compare_strategy_rewards import aggregate, validate, write_comparison


def trial(scale, seed, seconds, finished=True, reward=0):
    return {
        "time_scale_s": scale,
        "seed": seed,
        "elapsed_s": seconds,
        "finished": finished,
        "episode_done": "laps" if finished else "off_track",
        "pit_stops": 1,
        "return": reward,
        "report": f"scale_{scale}_seed_{seed}/evaluation/index.html",
    }


def test_comparison_uses_time_and_finish_rate_not_incomparable_rewards():
    rows = [
        trial(100, 0, 900, reward=6),
        trial(100, 1, 920, reward=5.8),
        trial(25, 0, 850, reward=-19),
        trial(25, 1, 870, reward=-19.8),
        trial(50, 0, 600),
        trial(50, 1, 50, finished=False),
    ]
    summary = aggregate(rows)
    assert [r["time_scale_s"] for r in summary] == [25, 100, 50]
    assert summary[0]["mean_s"] == 860
    assert summary[0]["std_s"] == pytest.approx(14.1421356)
    assert summary[-1]["mean_s"] == 600  # failure is not treated as a fast finish
    assert summary[-1]["finished"] == 1


def test_all_failures_have_no_time_estimate():
    summary = aggregate([trial(25, 0, 30, finished=False)])
    assert summary[0]["mean_s"] is None
    assert summary[0]["std_s"] is None
    assert summary[0]["mean_stops"] is None


def test_report_retains_seed_results_and_cautions(tmp_path):
    manifest = {"steps": 32, "laps": 8, "seeds": [0, 1], "scales": [100, 25]}
    trials = [trial(100, 0, 950), trial(25, 1, 900)]
    write_comparison(tmp_path, manifest, trials, {"<reference>": {"elapsed_s": 960, "finished": True}})
    data = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert len(data["trials"]) == 2
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "nicht vergleichbar" in page and "Pilotversuch" in page
    assert "&lt;reference&gt;" in page and "<reference>" not in page
    assert (tmp_path / "comparison.csv").exists()


@pytest.mark.parametrize(
    "scales,seeds,steps,rollout,laps,jobs",
    [
        ([0], [0], 32, 16, 8, 1),
        ([25, 25], [0], 32, 16, 8, 1),
        ([float("nan")], [0], 32, 16, 8, 1),
        ([25], [0, 0], 32, 16, 8, 1),
        ([25], [-1], 32, 16, 8, 1),
        ([25], [0], 33, 16, 8, 1),
        ([25], [0], 32, 1, 8, 1),
        ([25], [0], 32, 16, 1, 1),
        ([25], [0], 32, 16, 8, 0),
    ],
)
def test_unfair_or_invalid_experiments_rejected(scales, seeds, steps, rollout, laps, jobs):
    with pytest.raises(ValueError):
        validate(scales, seeds, steps, rollout, laps, jobs)
