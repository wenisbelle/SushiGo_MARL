import threading
import time
from io import StringIO

import pytest

from league.train_league import (
    PRESETS,
    build_run_specs,
    main,
    preflight,
    run_pending,
    run_training,
)


EXPECTED_TRAINING_ARGS = {
    "iterations": 5_000,
    "frames_per_batch": 5_000,
    "buffer_size": 100_000,
    "batch_size": 256,
    "updates_per_batch": 8,
    "num_workers": 12,
    "lr": 2.5e-4,
    "gamma": 0.99,
    "target_eps": 0.995,
    "max_grad_norm": 10.0,
    "reward_scale": 0.1,
    "mlp_cells": 128,
    "mlp_depth": 3,
}


def test_presets_match_dev_wenis_parameters():
    for player_count in (2, 3, 4):
        args = PRESETS[f"fixed_{player_count}p"].training_args()
        assert args.pop("n_players") == player_count
        assert args == EXPECTED_TRAINING_ARGS


def test_variable_preset_uses_shared_parameters():
    args = PRESETS["variable_2_4"].training_args()
    assert args.pop("min_n_players") == 2
    assert args.pop("max_n_players") == 4
    assert args == EXPECTED_TRAINING_ARGS


def test_default_matrix_has_twelve_unique_runs(tmp_path):
    specs = build_run_specs(list(PRESETS), 3, tmp_path, use_cuda=False)
    assert len(specs) == 12
    assert len({spec.run_dir for spec in specs}) == 12


def test_preflight_skips_complete_and_rejects_incomplete(tmp_path):
    specs = build_run_specs(["fixed_2p"], 2, tmp_path, use_cuda=False)
    specs[0].run_dir.mkdir(parents=True)
    specs[0].complete_path.touch()
    pending, completed = preflight(specs)
    assert pending == [specs[1]]
    assert completed == [specs[0]]

    specs[1].run_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Incomplete run directories"):
        preflight(specs)


def test_dry_run_has_no_filesystem_side_effects(tmp_path):
    output_root = tmp_path / "league"
    exit_code = main([
        "--dry-run",
        "--repetitions", "1",
        "--presets", "fixed_2p",
        "--output-root", str(output_root),
    ])
    assert exit_code == 0
    assert not output_root.exists()


def test_parallelism_is_bounded(tmp_path):
    specs = build_run_specs(["fixed_2p"], 5, tmp_path, use_cuda=False)
    lock = threading.Lock()
    active = 0
    peak = 0

    def runner(_spec):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return True

    assert run_pending(specs, parallelism=2, runner=runner)
    assert peak == 2


def test_successful_run_writes_artifacts_and_completion_marker(tmp_path, monkeypatch):
    spec = build_run_specs(["fixed_2p"], 1, tmp_path, use_cuda=False)[0]

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            self.stdout = StringIO("training complete\n")
            spec.model_path.write_bytes(b"model")
            spec.metrics_path.write_text("iteration,loss\n0,0.1\n", encoding="utf-8")
            assert "--num-workers" in command

        def wait(self):
            return 0

    monkeypatch.setattr(
        "league.train_league._git_metadata",
        lambda: {"commit": "test", "dirty": False},
    )
    monkeypatch.setattr("league.train_league._package_versions", lambda: {})
    monkeypatch.setattr("league.train_league.subprocess.Popen", FakeProcess)
    assert run_training(spec)
    assert spec.config_path.is_file()
    assert spec.stdout_path.read_text(encoding="utf-8") == "training complete\n"
    assert spec.complete_path.is_file()
