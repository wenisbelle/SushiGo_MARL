"""Run a granular two-player round-robin league over completed DQN checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys
from typing import Sequence

import torch


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from SushiGo_env.sushi_go_env import SushiGoParallelEnv
from league.policies import (
    ELIGIBLE_2P_PRESETS,
    LoadedPolicy,
    checkpoint_pairs,
    discover_checkpoints,
    load_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_ROOT = REPO_ROOT / "results" / "league"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "league_results" / "2p"
CSV_FIELDS = (
    "match_id",
    "matchup_id",
    "game_number",
    "competitor",
    "repetition",
    "checkpoint_path",
    "seat",
    "points",
    "pudding",
    "opponent_competitor",
    "opponent_repetition",
    "opponent_points",
    "opponent_pudding",
    "outcome",
)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-matchup", type=int, default=100)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--competitors",
        nargs="+",
        choices=ELIGIBLE_2P_PRESETS,
        default=list(ELIGIBLE_2P_PRESETS),
    )
    return parser


def outcomes(points: Sequence[int], puddings: Sequence[int]) -> tuple[str, str]:
    """Resolve points, then pudding, leaving an exact tie as a tie."""
    if points[0] != points[1]:
        winner = 0 if points[0] > points[1] else 1
    elif puddings[0] != puddings[1]:
        winner = 0 if puddings[0] > puddings[1] else 1
    else:
        return "tie", "tie"
    return ("win", "loss") if winner == 0 else ("loss", "win")


def play_game(
    policies: Sequence[LoadedPolicy],
    rng=random,
):
    """Play one unseeded game after independently randomizing checkpoint seats."""
    seated = rng.sample(list(policies), k=2)
    env = SushiGoParallelEnv(
        n_players=None,
        min_n_players=2,
        max_n_players=4,
        reward_scale=1.0,
    )
    observations, _ = env.reset(options={"n_players": 2})
    points = [0.0, 0.0]
    while env.agents:
        actions = {
            f"player_{seat}": policy.action(observations, seat)
            for seat, policy in enumerate(seated)
        }
        observations, rewards, _, _, _ = env.step(actions)
        for seat in range(2):
            points[seat] += rewards[f"player_{seat}"]

    integer_points = [int(round(value)) for value in points]
    puddings = [int(env.pudding_total[seat]) for seat in range(2)]
    game_outcomes = outcomes(integer_points, puddings)
    env.close()
    return seated, integer_points, puddings, game_outcomes


def result_rows(
    seated,
    points,
    puddings,
    game_outcomes,
    match_id,
    matchup_id,
    game_number,
):
    rows = []
    for seat in range(2):
        opponent = 1 - seat
        policy = seated[seat]
        opponent_policy = seated[opponent]
        rows.append({
            "match_id": match_id,
            "matchup_id": matchup_id,
            "game_number": game_number,
            "competitor": policy.spec.competitor,
            "repetition": policy.spec.repetition,
            "checkpoint_path": str(policy.spec.checkpoint_path.resolve()),
            "seat": seat,
            "points": points[seat],
            "pudding": puddings[seat],
            "opponent_competitor": opponent_policy.spec.competitor,
            "opponent_repetition": opponent_policy.spec.repetition,
            "opponent_points": points[opponent],
            "opponent_pudding": puddings[opponent],
            "outcome": game_outcomes[seat],
        })
    return rows


def run_league(policies, games_per_matchup: int, output_dir: Path, rng=random) -> int:
    pairs = checkpoint_pairs([policy.spec for policy in policies])
    policies_by_path = {policy.spec.checkpoint_path: policy for policy in policies}
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "matches.csv"
    total_games = len(pairs) * games_per_matchup
    completed_games = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for matchup_number, (left_spec, right_spec) in enumerate(pairs, start=1):
            matchup_id = f"matchup_{matchup_number:04d}"
            pair = (
                policies_by_path[left_spec.checkpoint_path],
                policies_by_path[right_spec.checkpoint_path],
            )
            print(
                f"[{matchup_number}/{len(pairs)}] {left_spec.label} vs "
                f"{right_spec.label} ({games_per_matchup} games)",
                flush=True,
            )
            progress_interval = max(1, games_per_matchup // 10)
            for game_number in range(1, games_per_matchup + 1):
                completed_games += 1
                match_id = f"match_{completed_games:06d}"
                game = play_game(pair, rng=rng)
                writer.writerows(result_rows(
                    *game,
                    match_id=match_id,
                    matchup_id=matchup_id,
                    game_number=game_number,
                ))
                csv_file.flush()
                if game_number % progress_interval == 0 or game_number == games_per_matchup:
                    print(
                        f"  game {game_number}/{games_per_matchup}; overall "
                        f"{completed_games}/{total_games}",
                        flush=True,
                    )
    return completed_games


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.games_per_matchup < 1:
        parser.error("--games-per-matchup must be at least 1")
    if args.output_dir.exists():
        parser.error(
            f"output directory already exists: {args.output_dir}. "
            "Delete it before rerunning."
        )

    specs = discover_checkpoints(args.models_root, args.competitors)
    if len(specs) < 2:
        parser.error("At least two eligible completed checkpoints are required")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device == "cpu":
        print("CUDA requested but unavailable; using CPU", file=sys.stderr)
    print(f"Loading {len(specs)} checkpoints on {device}...", flush=True)
    policies = [load_policy(spec, device=device) for spec in specs]
    n_pairs = len(checkpoint_pairs(specs))
    print(
        f"Running {n_pairs} matchups and {n_pairs * args.games_per_matchup} games",
        flush=True,
    )
    run_league(policies, args.games_per_matchup, args.output_dir)
    print(f"Results written to {args.output_dir / 'matches.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
