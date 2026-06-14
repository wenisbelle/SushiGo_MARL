import csv

from train.train_dqn import format_duration, log_progress_to_csv


def test_format_duration():
    assert format_duration(9.8) == "9s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3_726) == "1h 02m 06s"


def test_log_progress_includes_relative_time(tmp_path):
    path = tmp_path / "metrics.csv"
    log_progress_to_csv(
        filepath=path,
        iteration=4,
        elapsed_seconds=12.5,
        loss=0.25,
        epsilon=0.5,
        mean_reward=1.5,
        mean_return=3.0,
    )

    with path.open(newline="") as file:
        rows = list(csv.reader(file))

    assert rows == [
        [
            "iteration",
            "elapsed_seconds",
            "loss",
            "epsilon",
            "mean_turn_reward",
            "mean_episode_return",
        ],
        ["4", "12.5", "0.25", "0.5", "1.5", "3.0"],
    ]
