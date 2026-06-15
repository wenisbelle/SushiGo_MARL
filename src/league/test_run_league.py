import csv
import json
from pathlib import Path

import numpy as np
import pytest

from SushiGo_env.sushi_go_env import SushiGoParallelEnv
from SushiGo_env.torchrl_integration import GROUP
from league.policies import (
    CheckpointSpec,
    checkpoint_pairs,
    discover_checkpoints,
    load_policy,
    observations_to_tensordict,
)
from league.run_league import build_arg_parser, main, play_game, result_rows, run_league


REPO_ROOT = Path(__file__).resolve().parents[2]


def make_checkpoint(root, competitor, repetition, *, complete=True):
    run_dir = root / competitor / f"repetition_{repetition}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({
        "preset": competitor,
        "training_args": {"n_players": 2},
    }), encoding="utf-8")
    (run_dir / "model.pt").write_bytes(b"checkpoint")
    if complete:
        (run_dir / "COMPLETE").touch()
    return run_dir


def test_discovery_includes_only_completed_eligible_checkpoints(tmp_path):
    make_checkpoint(tmp_path, "fixed_2p", 2)
    make_checkpoint(tmp_path, "fixed_2p", 1)
    make_checkpoint(tmp_path, "variable_2_4", 1, complete=False)
    make_checkpoint(tmp_path, "fixed_3p", 1)

    specs = discover_checkpoints(tmp_path)
    assert [spec.label for spec in specs] == [
        "fixed_2p/repetition_1",
        "fixed_2p/repetition_2",
    ]


def test_completed_checkpoint_requires_config_and_model(tmp_path):
    run_dir = make_checkpoint(tmp_path, "fixed_2p", 1)
    (run_dir / "model.pt").unlink()
    with pytest.raises(RuntimeError, match="model.pt"):
        discover_checkpoints(tmp_path)


def test_checkpoint_pairs_include_cross_repetitions_without_self_pairs(tmp_path):
    specs = []
    for competitor, repetition in (
        ("fixed_2p", 1),
        ("fixed_2p", 2),
        ("variable_2_4", 1),
    ):
        run_dir = make_checkpoint(tmp_path, competitor, repetition)
        specs.append(CheckpointSpec(
            competitor, repetition, run_dir, run_dir / "model.pt", {}
        ))
    pairs = checkpoint_pairs(specs)
    assert len(pairs) == 3
    assert all(left != right for left, right in pairs)
    assert any(left.competitor == right.competitor for left, right in pairs)


def test_cli_defaults_to_100_games_and_accepts_override():
    parser = build_arg_parser()
    assert parser.parse_args([]).games_per_matchup == 100
    assert parser.parse_args(["--games-per-matchup", "7"]).games_per_matchup == 7


def test_fixed_and_variable_tensordicts_use_native_shapes_from_same_state():
    env = SushiGoParallelEnv(
        n_players=None, min_n_players=2, max_n_players=4
    )
    observations, _ = env.reset(options={"n_players": 2})
    fixed = observations_to_tensordict(observations, 2, "cpu")
    variable = observations_to_tensordict(observations, 4, "cpu")

    assert fixed[(GROUP, "observation", "hand_history")].shape == (2, 1, 12)
    assert fixed[(GROUP, "observation", "opponent_tableaus")].shape == (2, 1, 14)
    assert variable[(GROUP, "observation", "hand_history")].shape == (4, 3, 12)
    assert variable[(GROUP, "observation", "opponent_tableaus")].shape == (4, 3, 14)
    assert variable[(GROUP, "observation", "player_mask")].squeeze(-1).tolist() == [
        True, True, False, False
    ]


def test_result_rows_record_seats_and_reciprocal_results(tmp_path):
    class Policy:
        def __init__(self, competitor, repetition):
            model = tmp_path / competitor / f"repetition_{repetition}" / "model.pt"
            self.spec = CheckpointSpec(
                competitor, repetition, model.parent, model, {}
            )

    seated = [Policy("variable_2_4", 3), Policy("fixed_2p", 1)]
    rows = result_rows(
        seated, [42, 39], [4, 6], ("win", "loss"),
        match_id="match_000001", matchup_id="matchup_0001", game_number=1,
    )
    assert len(rows) == 2
    assert [row["seat"] for row in rows] == [0, 1]
    assert rows[0]["competitor"] == rows[1]["opponent_competitor"]
    assert rows[0]["repetition"] == rows[1]["opponent_repetition"]
    assert rows[0]["points"] == rows[1]["opponent_points"]
    assert rows[0]["pudding"] == rows[1]["opponent_pudding"]
    assert [row["outcome"] for row in rows] == ["win", "loss"]


def test_play_game_uses_randomized_seat_assignment(tmp_path):
    class FirstLegalPolicy:
        def __init__(self, competitor):
            model = tmp_path / competitor / "repetition_1" / "model.pt"
            self.spec = CheckpointSpec(competitor, 1, model.parent, model, {})

        def action(self, observations, seat):
            return int(np.flatnonzero(
                observations[f"player_{seat}"]["action_mask"]
            )[0])

    class ReverseRng:
        def __init__(self):
            self.calls = 0

        def sample(self, policies, k):
            self.calls += 1
            assert k == 2
            return list(reversed(policies))

    policies = [FirstLegalPolicy("fixed_2p"), FirstLegalPolicy("variable_2_4")]
    rng = ReverseRng()
    seated, points, puddings, game_outcomes = play_game(policies, rng=rng)
    assert rng.calls == 1
    assert [policy.spec.competitor for policy in seated] == [
        "variable_2_4", "fixed_2p"
    ]
    assert len(points) == len(puddings) == len(game_outcomes) == 2


def test_run_league_writes_two_rows_per_game(tmp_path, monkeypatch):
    class Policy:
        def __init__(self, competitor):
            model = tmp_path / competitor / "repetition_1" / "model.pt"
            self.spec = CheckpointSpec(competitor, 1, model.parent, model, {})

    policies = [Policy("fixed_2p"), Policy("variable_2_4")]

    def fake_game(pair, rng):
        assert rng is np.random
        return list(reversed(pair)), [30, 30], [5, 5], ("tie", "tie")

    monkeypatch.setattr("league.run_league.play_game", fake_game)
    output_dir = tmp_path / "results"
    assert run_league(policies, 3, output_dir, rng=np.random) == 3
    with (output_dir / "matches.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 6
    for first, second in zip(rows[::2], rows[1::2]):
        assert first["match_id"] == second["match_id"]
        assert first["matchup_id"] == second["matchup_id"]
        assert {first["seat"], second["seat"]} == {"0", "1"}
        assert first["competitor"] == second["opponent_competitor"]


def test_existing_output_directory_fails_before_checkpoint_loading(tmp_path, monkeypatch):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    monkeypatch.setattr(
        "league.run_league.discover_checkpoints",
        lambda *_args, **_kwargs: pytest.fail("discovery should not run"),
    )
    with pytest.raises(SystemExit):
        main(["--output-dir", str(output_dir)])


SAMPLE_ROOT = REPO_ROOT / "results" / "league"
SAMPLE_MODELS_AVAILABLE = all(
    (SAMPLE_ROOT / preset / "repetition_1" / "model.pt").is_file()
    for preset in ("fixed_2p", "variable_2_4", "variable_encoder_2_4")
)


@pytest.mark.skipif(not SAMPLE_MODELS_AVAILABLE, reason="sample league models unavailable")
def test_loads_and_runs_all_current_eligible_sample_checkpoints():
    specs = discover_checkpoints(SAMPLE_ROOT)
    assert {spec.competitor for spec in specs} == {
        "fixed_2p", "variable_2_4", "variable_encoder_2_4"
    }
    env = SushiGoParallelEnv(
        n_players=None, min_n_players=2, max_n_players=4
    )
    observations, _ = env.reset(options={"n_players": 2})
    for policy in (load_policy(spec) for spec in specs):
        action = policy.action(observations, seat=0)
        assert observations["player_0"]["action_mask"][action]
