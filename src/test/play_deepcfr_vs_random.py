"""Run Sushi Go matches between trained Deep CFR and random policies.

This script evaluates the Deep CFR policy against a random agent baseline.
By default it pits the Deep CFR checkpoint against a random player in a 2-player
game and prints the final scores.

Example:
    python3 test/play_deepcfr_vs_random.py \
        --n-players 2 \
        --deepcfr-model models/DeepCFR/sushi_go_deep_cfr_policy_2_players.pt

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


POLICY_DEEPCFR = "deepcfr"
POLICY_RANDOM = "random"


def load_deepcfr_model(obs_dim: int, path: str):
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
        print(f"[model] loaded Deep CFR checkpoint: {path}")
        return model
    except FileNotFoundError:
        print(f"[model] Deep CFR checkpoint not found: {path} -> seat will play randomly")
    except Exception as exc:  # noqa: BLE001
        print(f"[model] failed to load Deep CFR checkpoint: {path} ({exc}) -> seat will play randomly")
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
        POLICY_DEEPCFR: "DeepCFR",
        POLICY_RANDOM: "Random",
    }.get(policy_name, policy_name)


def determine_winners(env, totals):
    best_score = max(totals.values())
    tied_agents = [i for i, agent in enumerate(env.possible_agents) if totals[agent] == best_score]
    if len(tied_agents) > 1:
        best_pudding = max(int(env.pudding_total[i]) for i in tied_agents)
        tied_agents = [i for i in tied_agents if int(env.pudding_total[i]) == best_pudding]
    return tied_agents


def run_game(env, seat_policies, deepcfr_model, rng, stochastic_deepcfr, verbose=False, seed=None):
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

            if policy == POLICY_DEEPCFR and deepcfr_model is not None:
                action = deepcfr_action(deepcfr_model, obs, mask, rng, stochastic=stochastic_deepcfr)
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


def build_seat_policies(n_players, deepcfr_seat):
    policies = [POLICY_RANDOM for _ in range(n_players)]
    policies[deepcfr_seat] = POLICY_DEEPCFR
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
    parser = argparse.ArgumentParser(description="Run Deep CFR vs Random matches in Sushi Go")
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
        "--deepcfr-model",
        type=str,
        default="models/DeepCFR/sushi_go_deep_cfr_policy_2_players.pt",
    )
    parser.add_argument(
        "--deepcfr-seat",
        type=int,
        default=0,
        help="seat index controlled by Deep CFR",
    )
    parser.add_argument("--deepcfr-greedy", action="store_true", help="use argmax for Deep CFR instead of sampling")
    parser.add_argument("--verbose", action="store_true", help="print each turn's actions")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0 <= args.deepcfr_seat < args.n_players):
        raise ValueError("--deepcfr-seat must be within the player range")

    rng = np.random.default_rng(args.seed)
    env = SushiGoParallelEnv(n_players=args.n_players, reward_scale=1.0)
    observations, _ = env.reset(seed=args.seed)
    obs_dim = observations[env.possible_agents[0]]["observation"].shape[-1]

    deepcfr_model = load_deepcfr_model(obs_dim, args.deepcfr_model)

    seat_policies = build_seat_policies(args.n_players, args.deepcfr_seat)
    stochastic_deepcfr = not args.deepcfr_greedy

    print("\nMatch setup")
    print(f"  players: {args.n_players}")
    print(f"  DeepCFR seat: {args.deepcfr_seat}")
    print(f"  policies: {', '.join(seat_label(p) for p in seat_policies)}")

    wins = {"deepcfr": 0, "random": 0, "tie": 0}
    deepcfr_scores = []
    random_scores = []
    victory_margins = []
    
    for game_idx in range(1, args.games + 1):
        env_seed = None if args.seed is None else args.seed + game_idx - 1

        totals, winners = run_game(
            env=env,
            seat_policies=seat_policies,
            deepcfr_model=deepcfr_model,
            rng=rng,
            stochastic_deepcfr=stochastic_deepcfr,
            verbose=args.verbose,
            seed=env_seed,
        )

        # Track scores for each agent
        deepcfr_agent = env.possible_agents[args.deepcfr_seat]
        random_seat = 1 - args.deepcfr_seat if args.n_players == 2 else (0 if args.deepcfr_seat != 0 else 1)
        random_agent = env.possible_agents[random_seat]
        deepcfr_score = totals[deepcfr_agent]
        random_score = totals[random_agent]
        deepcfr_scores.append(deepcfr_score)
        random_scores.append(random_score)

        if len(winners) == 1:
            winner_seat = winners[0]
            winner_policy = seat_policies[winner_seat]
            if winner_policy == POLICY_DEEPCFR:
                wins["deepcfr"] += 1
                victory_margins.append(deepcfr_score - random_score)
            elif winner_policy == POLICY_RANDOM:
                wins["random"] += 1
                victory_margins.append(random_score - deepcfr_score)
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
        print(f"  DeepCFR wins: {wins['deepcfr']}")
        print(f"  Random wins: {wins['random']}")
        print(f"  ties: {wins['tie']}")
        
        # Calculate statistics
        deepcfr_win_rate = (wins["deepcfr"] / args.games) * 100
        random_win_rate = (wins["random"] / args.games) * 100
        deepcfr_avg_score = np.mean(deepcfr_scores)
        random_avg_score = np.mean(random_scores)
        
        # Calculate average margin (only for non-tie games)
        non_tie_margins = [m for m in victory_margins if m != 0]
        avg_margin = np.mean(non_tie_margins) if non_tie_margins else 0.0
        
        print("\nStatistics")
        print(f"  DeepCFR win rate: {deepcfr_win_rate:.1f}%")
        print(f"  Random win rate: {random_win_rate:.1f}%")
        print(f"  Average score DeepCFR: {deepcfr_avg_score:.1f}")
        print(f"  Average score Random: {random_avg_score:.1f}")
        print(f"  Average victory margin: {avg_margin:.1f}")


if __name__ == "__main__":
    main()
