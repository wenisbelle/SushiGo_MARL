"""Run Sushi Go matches between two trained Deep CFR policies.

This script compares two versions of Deep CFR policies by pitting them against
each other in competitive matches.

Example:
    python3 test/play_deepcfr_vs_deepcfr.py \
        --n-players 2 \
        --deepcfr-model-1 models/DeepCFR/sushi_go_deep_cfr_policy_2_players.pt \
        --deepcfr-model-2 models/DeepCFR/sushi_go_deep_cfr_policy_2_players.pt

If a checkpoint cannot be loaded, that seat falls back to a random legal move so
the match can still run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SushiGo_env"))

from SushiGo_env.sushi_go_env import CARD_NAMES, N_TYPES, SushiGoParallelEnv  # noqa: E402
from train.train_dcfr_v2 import MLP  # noqa: E402


POLICY_DEEPCFR_1 = "deepcfr_1"
POLICY_DEEPCFR_2 = "deepcfr_2"
POLICY_RANDOM = "random"


def load_deepcfr_model(obs_dim: int, path: str, model_name: str = "Deep CFR"):
    """Load the Deep CFR average-policy network saved by train_deep_cfr.py."""
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu")
        hidden_dim = int(checkpoint.get("hidden_dim", 64)) if isinstance(checkpoint, dict) else 64
        depth = int(checkpoint.get("depth", 2)) if isinstance(checkpoint, dict) else 2

        model = MLP(obs_dim, N_TYPES, hidden_dim=hidden_dim, depth=depth)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        print(f"[model] loaded {model_name} checkpoint: {path}")
        return model
    except FileNotFoundError:
        print(f"[model] {model_name} checkpoint not found: {path} -> seat will play randomly")
    except Exception as exc:  # noqa: BLE001
        print(f"[model] failed to load {model_name} checkpoint: {path} ({exc}) -> seat will play randomly")
    return None


def deepcfr_action(model, obs, mask, rng, stochastic=True):
    """Masked Deep CFR action for a single seat."""
    import torch

    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        logits = model(obs_t)
        masked_logits = logits.masked_fill(~mask_t, -1e9)
        if stochastic:
            probs = torch.softmax(masked_logits, dim=-1)
            probs = probs * mask_t.float()
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return int(torch.multinomial(probs.squeeze(0), num_samples=1).item())
        return int(torch.argmax(masked_logits, dim=-1).item())


def random_action(mask, rng):
    """Random legal action."""
    legal = np.flatnonzero(mask.astype(bool))
    return int(rng.choice(legal))


def seat_label(policy_name):
    return {
        POLICY_DEEPCFR_1: "DeepCFR-v1",
        POLICY_DEEPCFR_2: "DeepCFR-v2",
        POLICY_RANDOM: "Random",
    }.get(policy_name, policy_name)


def determine_winners(env, totals):
    best_score = max(totals.values())
    tied_agents = [i for i, agent in enumerate(env.possible_agents) if totals[agent] == best_score]
    if len(tied_agents) > 1:
        best_pudding = max(int(env.pudding_total[i]) for i in tied_agents)
        tied_agents = [i for i in tied_agents if int(env.pudding_total[i]) == best_pudding]
    return tied_agents


def run_game(env, seat_policies, deepcfr_models, rng, stochastic_deepcfr, verbose=False, seed=None):
    observations, _ = env.reset(seed=seed)
    totals = {agent: 0.0 for agent in env.possible_agents}
    turn = 0

    while env.agents:
        active_agents = list(env.agents)
        actions = {}
        turn_log = []

        for seat_idx, agent in enumerate(active_agents):
            policy = seat_policies[seat_idx]
            obs = observations[agent]["observation"]
            mask = observations[agent]["action_mask"]

            if policy == POLICY_DEEPCFR_1 and deepcfr_models[0] is not None:
                action = deepcfr_action(deepcfr_models[0], obs, mask, rng, stochastic=stochastic_deepcfr)
            elif policy == POLICY_DEEPCFR_2 and deepcfr_models[1] is not None:
                action = deepcfr_action(deepcfr_models[1], obs, mask, rng, stochastic=stochastic_deepcfr)
            else:
                action = random_action(mask, rng)

            actions[agent] = action
            if verbose:
                turn_log.append(f"{agent}:{seat_label(policy)}->{CARD_NAMES[action]}")

        observations, rewards, _, _, _ = env.step(actions)
        for agent, reward in rewards.items():
            totals[agent] += float(reward)

        turn += 1
        if verbose:
            print(f"turn {turn:02d} | " + " | ".join(turn_log))

    winners = determine_winners(env, totals)
    return totals, winners


def build_seat_policies(n_players, deepcfr_1_seat, deepcfr_2_seat):
    policies = [POLICY_RANDOM for _ in range(n_players)]
    policies[deepcfr_1_seat] = POLICY_DEEPCFR_1
    policies[deepcfr_2_seat] = POLICY_DEEPCFR_2
    return policies


def format_scoreboard(env, totals, seat_policies):
    rows = []
    for idx, agent in enumerate(env.possible_agents):
        rows.append(
            f"seat {idx} [{seat_label(seat_policies[idx])}]: "
            f"{totals[agent]:5.1f} pts | pudding={int(env.pudding_total[idx])}"
        )
    return "\n".join(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep CFR vs Deep CFR matches in Sushi Go")
    parser.add_argument("--n-players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument(
        "--n-games",
        "--games",
        dest="games",
        type=int,
        default=1000,
        help="number of matches to run",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deepcfr-model-1",
        type=str,
        default="models/DeepCFR/sushi_go_deep_cfr_policy_2_players.pt",
        help="path to first Deep CFR model",
    )
    parser.add_argument(
        "--deepcfr-model-2",
        type=str,
        default="models/DFCR_old/sushi_go_deep_cfr_policy_2_players.pt",
        help="path to second Deep CFR model",
    )
    parser.add_argument(
        "--deepcfr-1-seat",
        type=int,
        default=0,
        help="seat index controlled by first Deep CFR model",
    )
    parser.add_argument(
        "--deepcfr-2-seat",
        type=int,
        default=1,
        help="seat index controlled by second Deep CFR model",
    )
    parser.add_argument("--deepcfr-greedy", action="store_true", help="use argmax for Deep CFR instead of sampling")
    parser.add_argument("--verbose", action="store_true", help="print each turn's actions")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0 <= args.deepcfr_1_seat < args.n_players):
        raise ValueError("--deepcfr-1-seat must be within the player range")
    if not (0 <= args.deepcfr_2_seat < args.n_players):
        raise ValueError("--deepcfr-2-seat must be within the player range")
    if args.deepcfr_1_seat == args.deepcfr_2_seat:
        raise ValueError("DeepCFR-v1 and DeepCFR-v2 cannot control the same seat")

    rng = np.random.default_rng(args.seed)
    env = SushiGoParallelEnv(n_players=args.n_players, reward_scale=1.0)
    observations, _ = env.reset(seed=args.seed)
    obs_dim = observations[env.possible_agents[0]]["observation"].shape[-1]

    deepcfr_model_1 = load_deepcfr_model(obs_dim, args.deepcfr_model_1, "DeepCFR-v1")
    deepcfr_model_2 = load_deepcfr_model(obs_dim, args.deepcfr_model_2, "DeepCFR-v2")

    seat_policies = build_seat_policies(args.n_players, args.deepcfr_1_seat, args.deepcfr_2_seat)
    stochastic_deepcfr = not args.deepcfr_greedy

    print("\nMatch setup")
    print(f"  players: {args.n_players}")
    print(f"  DeepCFR-v1 seat: {args.deepcfr_1_seat}")
    print(f"  DeepCFR-v2 seat: {args.deepcfr_2_seat}")
    print(f"  policies: {', '.join(seat_label(p) for p in seat_policies)}")

    wins = {"deepcfr_1": 0, "deepcfr_2": 0, "tie": 0}
    deepcfr_1_scores = []
    deepcfr_2_scores = []
    victory_margins = []
    
    for game_idx in range(1, args.games + 1):
        env_seed = None if args.seed is None else args.seed + game_idx - 1

        totals, winners = run_game(
            env=env,
            seat_policies=seat_policies,
            deepcfr_models=[deepcfr_model_1, deepcfr_model_2],
            rng=rng,
            stochastic_deepcfr=stochastic_deepcfr,
            verbose=args.verbose,
            seed=env_seed,
        )

        # Track scores for each agent
        deepcfr_1_agent = env.possible_agents[args.deepcfr_1_seat]
        deepcfr_2_agent = env.possible_agents[args.deepcfr_2_seat]
        deepcfr_1_score = totals[deepcfr_1_agent]
        deepcfr_2_score = totals[deepcfr_2_agent]
        deepcfr_1_scores.append(deepcfr_1_score)
        deepcfr_2_scores.append(deepcfr_2_score)

        if len(winners) == 1:
            winner_seat = winners[0]
            winner_policy = seat_policies[winner_seat]
            if winner_policy == POLICY_DEEPCFR_1:
                wins["deepcfr_1"] += 1
                victory_margins.append(deepcfr_1_score - deepcfr_2_score)
            elif winner_policy == POLICY_DEEPCFR_2:
                wins["deepcfr_2"] += 1
                victory_margins.append(deepcfr_2_score - deepcfr_1_score)
        else:
            wins["tie"] += 1
            victory_margins.append(0)

        print(f"\nGame {game_idx}/{args.games}")
        print(format_scoreboard(env, totals, seat_policies))
        if len(winners) == 1:
            winner_seat = winners[0]
            print(f"winner: seat {winner_seat} [{seat_label(seat_policies[winner_seat])}]")
        else:
            winner_text = ", ".join(f"seat {seat}" for seat in winners)
            print(f"winner: tie between {winner_text}")

    if args.games > 1:
        print("\nSummary")
        print(f"  DeepCFR-v1 wins: {wins['deepcfr_1']}")
        print(f"  DeepCFR-v2 wins: {wins['deepcfr_2']}")
        print(f"  ties: {wins['tie']}")
        
        # Calculate statistics
        deepcfr_1_win_rate = (wins["deepcfr_1"] / args.games) * 100
        deepcfr_2_win_rate = (wins["deepcfr_2"] / args.games) * 100
        deepcfr_1_avg_score = np.mean(deepcfr_1_scores)
        deepcfr_2_avg_score = np.mean(deepcfr_2_scores)
        
        # Calculate average margin (only for non-tie games)
        non_tie_margins = [m for m in victory_margins if m != 0]
        avg_margin = np.mean(non_tie_margins) if non_tie_margins else 0.0
        
        print("\nStatistics")
        print(f"  DeepCFR-v1 win rate: {deepcfr_1_win_rate:.1f}%")
        print(f"  DeepCFR-v2 win rate: {deepcfr_2_win_rate:.1f}%")
        print(f"  Average score DeepCFR-v1: {deepcfr_1_avg_score:.1f}")
        print(f"  Average score DeepCFR-v2: {deepcfr_2_avg_score:.1f}")
        print(f"  Average victory margin: {avg_margin:.1f}")


if __name__ == "__main__":
    main()
