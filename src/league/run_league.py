"""Run granular round-robin leagues over completed DQN checkpoints."""

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
    ELIGIBLE_PRESETS_BY_PLAYERS,
    LoadedPolicy,
    checkpoint_matchups,
    discover_checkpoints,
    load_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_ROOT = REPO_ROOT / "results" / "league"
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
    "opponent_competitors",
    "opponent_repetitions",
    "opponent_points_all",
    "opponent_puddings",
    "outcome",
)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, choices=[2, 3, 4], required=True)
    parser.add_argument("--games-per-matchup", type=int, default=100)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--competitors",
        nargs="+",
        choices=tuple(dict.fromkeys(
            preset
            for presets in ELIGIBLE_PRESETS_BY_PLAYERS.values()
            for preset in presets
        )),
        default=None,
    )
    return parser


def outcomes(points: Sequence[int], puddings: Sequence[int]) -> tuple[str, ...]:
    """Resolve winner(s) by points, then pudding among point-tied leaders."""
    best_points = max(points)
    point_leaders = [i for i, value in enumerate(points) if value == best_points]
    if len(point_leaders) == 1:
        winners = point_leaders
    else:
        best_pudding = max(puddings[i] for i in point_leaders)
        winners = [i for i in point_leaders if puddings[i] == best_pudding]
    if len(winners) > 1:
        return tuple("tie" if i in winners else "loss" for i in range(len(points)))
    return tuple("win" if i == winners[0] else "loss" for i in range(len(points)))


def play_game(
    policies: Sequence[LoadedPolicy],
    players: int = 2,
    rng=random,
):
    """Play one unseeded game after independently randomizing checkpoint seats."""
    seated = rng.sample(list(policies), k=players)
    env = SushiGoParallelEnv(
        n_players=None,
        min_n_players=2,
        max_n_players=4,
        reward_scale=1.0,
    )
    observations, _ = env.reset(options={"n_players": players})
    points = [0.0 for _ in range(players)]
    while env.agents:
        actions = {
            f"player_{seat}": policy.action(observations, seat)
            for seat, policy in enumerate(seated)
        }
        observations, rewards, _, _, _ = env.step(actions)
        for seat in range(players):
            points[seat] += rewards[f"player_{seat}"]

    integer_points = [int(round(value)) for value in points]
    puddings = [int(env.pudding_total[seat]) for seat in range(players)]
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
    for seat in range(len(seated)):
        opponent_seats = [i for i in range(len(seated)) if i != seat]
        policy = seated[seat]
        first_opponent = opponent_seats[0]
        first_opponent_policy = seated[first_opponent]
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
            "opponent_competitor": first_opponent_policy.spec.competitor,
            "opponent_repetition": first_opponent_policy.spec.repetition,
            "opponent_points": points[first_opponent],
            "opponent_pudding": puddings[first_opponent],
            "opponent_competitors": "|".join(
                seated[i].spec.competitor for i in opponent_seats
            ),
            "opponent_repetitions": "|".join(
                str(seated[i].spec.repetition) for i in opponent_seats
            ),
            "opponent_points_all": "|".join(str(points[i]) for i in opponent_seats),
            "opponent_puddings": "|".join(str(puddings[i]) for i in opponent_seats),
            "outcome": game_outcomes[seat],
        })
    return rows


def run_league(
    policies,
    games_per_matchup: int,
    output_dir: Path,
    players: int = 2,
    rng=random,
) -> int:
    matchups = checkpoint_matchups([policy.spec for policy in policies], players)
    policies_by_path = {policy.spec.checkpoint_path: policy for policy in policies}
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "matches.csv"
    total_games = len(matchups) * games_per_matchup
    completed_games = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for matchup_number, matchup_specs in enumerate(matchups, start=1):
            matchup_id = f"matchup_{matchup_number:04d}"
            matchup = tuple(
                policies_by_path[spec.checkpoint_path]
                for spec in matchup_specs
            )
            matchup_label = " vs ".join(spec.label for spec in matchup_specs)
            print(
                f"[{matchup_number}/{len(matchups)}] {matchup_label} "
                f"({games_per_matchup} games)",
                flush=True,
            )
            progress_interval = max(1, games_per_matchup // 10)
            for game_number in range(1, games_per_matchup + 1):
                completed_games += 1
                match_id = f"match_{completed_games:06d}"
                game = play_game(matchup, players=players, rng=rng)
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
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else REPO_ROOT / "results" / "league_results" / f"{args.players}p"
    )
    competitors = (
        args.competitors
        if args.competitors is not None
        else list(ELIGIBLE_PRESETS_BY_PLAYERS[args.players])
    )
    if args.games_per_matchup < 1:
        parser.error("--games-per-matchup must be at least 1")
    if output_dir.exists():
        parser.error(
            f"output directory already exists: {output_dir}. "
            "Delete it before rerunning."
        )

    specs = discover_checkpoints(args.models_root, competitors)
    if len(specs) < args.players:
        parser.error(f"At least {args.players} eligible completed checkpoints are required")
    matchups = checkpoint_matchups(specs, args.players)
    if not matchups:
        parser.error("No valid matchups after excluding all-same-model tables")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device == "cpu":
        print("CUDA requested but unavailable; using CPU", file=sys.stderr)
    print(f"Loading {len(specs)} checkpoints on {device}...", flush=True)
    policies = [load_policy(spec, device=device) for spec in specs]
    print(
        f"Running {len(matchups)} matchups and {len(matchups) * args.games_per_matchup} games",
        flush=True,
    )
    run_league(policies, args.games_per_matchup, output_dir, players=args.players)
    print(f"Results written to {output_dir / 'matches.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
