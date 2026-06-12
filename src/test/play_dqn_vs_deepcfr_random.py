"""Run Sushi Go matches between 3-player DQN, Deep CFR, and a random seat.

This script evaluates a 3-player setup where one seat is controlled by a DQN
checkpoint, one seat is controlled by a Deep CFR checkpoint, and the remaining
seat plays random legal moves.

Example:
    python3 test/play_dqn_vs_deepcfr_random.py \
        --dqn-model models/DQN/sushi_go_qnet_3_players.pt \
        --deepcfr-model models/DeepCFR/sushi_go_deep_cfr_policy_3_players.pt

If either checkpoint cannot be loaded, that seat falls back to a random legal
move so the match can still run.
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
from tensordict import TensorDict  # noqa: E402

from SushiGo_env.torchrl_integration import ACTION_KEY, MASK_KEY, OBS_KEY  # noqa: E402
from train.train_dqn import build_qvalue_actor  # noqa: E402
from train.train_dcfr_v2 import MLP  # noqa: E402


POLICY_DQN = "dqn"
POLICY_DEEPCFR = "deepcfr"
POLICY_RANDOM = "random"


def load_dqn_model(n_players: int, obs_dim: int, path: str):
    """Load the shared Q-network saved by train_dqn.py."""
    import torch

    try:
        actor = build_qvalue_actor(n_players, obs_dim, device="cpu")
        dummy = TensorDict(
            {
                OBS_KEY: torch.zeros(n_players, obs_dim),
                MASK_KEY: torch.ones(n_players, N_TYPES, dtype=torch.bool),
            },
            batch_size=[],
        )
        actor(dummy)

        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        actor.load_state_dict(state_dict)
        actor.eval()
        print(f"[model] loaded DQN checkpoint: {path}")
        return actor
    except FileNotFoundError:
        print(f"[model] DQN checkpoint not found: {path} -> seat will play randomly")
    except Exception as exc:  # noqa: BLE001
        print(f"[model] failed to load DQN checkpoint: {path} ({exc}) -> seat will play randomly")
    return None


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


def dqn_actions(actor, obs_dict, agents):
    """Masked greedy actions for the active agents."""
    import torch

    obs = np.stack([obs_dict[a]["observation"] for a in agents]).astype(np.float32)
    mask = np.stack([obs_dict[a]["action_mask"] for a in agents]).astype(bool)
    td = TensorDict(
        {
            OBS_KEY: torch.from_numpy(obs),
            MASK_KEY: torch.from_numpy(mask),
        },
        batch_size=[],
    )
    with torch.no_grad():
        actor(td)
    return td[ACTION_KEY].cpu().numpy().astype(int)


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
    legal = np.flatnonzero(mask.astype(bool))
    return int(rng.choice(legal))


def seat_label(policy_name):
    return {
        POLICY_DQN: "DQN",
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


def run_game(env, seat_policies, dqn_actor, deepcfr_model, rng, stochastic_deepcfr, verbose=False, seed=None):
    observations, _ = env.reset(seed=seed)
    totals = {agent: 0.0 for agent in env.possible_agents}
    turn = 0

    while env.agents:
        active_agents = list(env.agents)
        dqn_batch = None
        if dqn_actor is not None and any(policy == POLICY_DQN for policy in seat_policies):
            dqn_batch = dqn_actions(dqn_actor, observations, active_agents)

        actions = {}
        turn_log = []
        for seat_idx, agent in enumerate(active_agents):
            policy = seat_policies[seat_idx]
            obs = observations[agent]["observation"]
            mask = observations[agent]["action_mask"]

            if policy == POLICY_DQN and dqn_batch is not None:
                action = int(dqn_batch[seat_idx])
            elif policy == POLICY_DEEPCFR and deepcfr_model is not None:
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


def build_seat_policies(dqn_seat, deepcfr_seat):
    policies = [POLICY_RANDOM for _ in range(3)]
    policies[dqn_seat] = POLICY_DQN
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
    parser = argparse.ArgumentParser(description="Run DQN vs Deep CFR with a random third seat")
    parser.add_argument("--n-games", "--games", dest="games", type=int, default=1000, help="number of matches to run")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--dqn-model", type=str, default="models/DQN/sushi_go_qnet_3_players.pt")
    parser.add_argument(
        "--deepcfr-model",
        type=str,
        default="models/DeepCFR/sushi_go_deep_cfr_policy_3_players.pt",
    )
    parser.add_argument("--dqn-seat", type=int, default=0, help="seat index controlled by DQN")
    parser.add_argument("--deepcfr-seat", type=int, default=1, help="seat index controlled by Deep CFR")
    parser.add_argument("--deepcfr-greedy", action="store_true", help="use argmax for Deep CFR instead of sampling")
    parser.add_argument("--verbose", action="store_true", help="print each turn's actions")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dqn_seat not in (0, 1, 2):
        raise ValueError("--dqn-seat must be 0, 1, or 2")
    if args.deepcfr_seat not in (0, 1, 2):
        raise ValueError("--deepcfr-seat must be 0, 1, or 2")
    if args.dqn_seat == args.deepcfr_seat:
        raise ValueError("DQN and Deep CFR cannot control the same seat")

    random_seat = next(seat for seat in range(3) if seat not in (args.dqn_seat, args.deepcfr_seat))

    rng = np.random.default_rng(args.seed)
    env = SushiGoParallelEnv(n_players=3, reward_scale=1.0)
    observations, _ = env.reset(seed=args.seed)
    obs_dim = observations[env.possible_agents[0]]["observation"].shape[-1]

    dqn_actor = load_dqn_model(3, obs_dim, args.dqn_model)
    deepcfr_model = load_deepcfr_model(obs_dim, args.deepcfr_model)

    seat_policies = build_seat_policies(args.dqn_seat, args.deepcfr_seat)
    stochastic_deepcfr = not args.deepcfr_greedy

    print("\nMatch setup")
    print("  players: 3")
    print(f"  DQN seat: {args.dqn_seat}")
    print(f"  DeepCFR seat: {args.deepcfr_seat}")
    print(f"  Random seat: {random_seat}")
    print(f"  policies: {', '.join(seat_label(p) for p in seat_policies)}")

    wins = {"dqn": 0, "deepcfr": 0, "random": 0, "tie": 0}
    dqn_scores = []
    deepcfr_scores = []
    random_scores = []
    victory_margins = []

    for game_idx in range(1, args.games + 1):
        env_seed = None if args.seed is None else args.seed + game_idx - 1

        totals, winners = run_game(
            env=env,
            seat_policies=seat_policies,
            dqn_actor=dqn_actor,
            deepcfr_model=deepcfr_model,
            rng=rng,
            stochastic_deepcfr=stochastic_deepcfr,
            verbose=args.verbose,
            seed=env_seed,
        )

        dqn_agent = env.possible_agents[args.dqn_seat]
        deepcfr_agent = env.possible_agents[args.deepcfr_seat]
        random_agent = env.possible_agents[random_seat]
        dqn_score = totals[dqn_agent]
        deepcfr_score = totals[deepcfr_agent]
        random_score = totals[random_agent]
        dqn_scores.append(dqn_score)
        deepcfr_scores.append(deepcfr_score)
        random_scores.append(random_score)

        if len(winners) == 1:
            winner_seat = winners[0]
            winner_policy = seat_policies[winner_seat]
            if winner_policy == POLICY_DQN:
                wins["dqn"] += 1
                victory_margins.append(dqn_score - max(deepcfr_score, random_score))
            elif winner_policy == POLICY_DEEPCFR:
                wins["deepcfr"] += 1
                victory_margins.append(deepcfr_score - max(dqn_score, random_score))
            else:
                wins["random"] += 1
                victory_margins.append(random_score - max(dqn_score, deepcfr_score))
        else:
            wins["tie"] += 1
            victory_margins.append(0.0)

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
        print(f"  DQN wins: {wins['dqn']}")
        print(f"  Deep CFR wins: {wins['deepcfr']}")
        print(f"  Random wins: {wins['random']}")
        print(f"  ties: {wins['tie']}")

        dqn_win_rate = (wins["dqn"] / args.games) * 100
        deepcfr_win_rate = (wins["deepcfr"] / args.games) * 100
        random_win_rate = (wins["random"] / args.games) * 100
        dqn_avg_score = np.mean(dqn_scores)
        deepcfr_avg_score = np.mean(deepcfr_scores)
        random_avg_score = np.mean(random_scores)
        non_tie_margins = [m for m in victory_margins if m != 0]
        avg_margin = np.mean(non_tie_margins) if non_tie_margins else 0.0

        print("\nStatistics")
        print(f"  DQN win rate: {dqn_win_rate:.1f}%")
        print(f"  Deep CFR win rate: {deepcfr_win_rate:.1f}%")
        print(f"  Random win rate: {random_win_rate:.1f}%")
        print(f"  Average score DQN: {dqn_avg_score:.1f}")
        print(f"  Average score Deep CFR: {deepcfr_avg_score:.1f}")
        print(f"  Average score Random: {random_avg_score:.1f}")
        print(f"  Average victory margin: {avg_margin:.1f}")


if __name__ == "__main__":
    main()
