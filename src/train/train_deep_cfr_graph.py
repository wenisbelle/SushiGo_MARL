"""
Deep CFR for Sushi Go! with the PettingZoo parallel environment.

This implementation keeps the project conventions already used by train_dqn.py:
- one shared network across seats
- legal-action masking from the environment
- one environment per player count

The algorithm here is a practical Deep CFR variant for this repository:
- an advantage network learns counterfactual action values
- an average-policy network learns the running strategy mixture
- reservoir buffers store samples across traversals
- action values are estimated by branching from the current state and rolling out
  the rest of the episode under the current policy

The code is intentionally self-contained so it can run without adding a new RL
stack or refactoring the environment.
"""
import argparse
import copy
from email import parser
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../SushiGo_env")))

from SushiGo_env.sushi_go_env import N_TYPES, SushiGoParallelEnv


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
	def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, depth: int):
		super().__init__()
		layers = []
		last_dim = input_dim
		for _ in range(depth):
			layers.append(nn.Linear(last_dim, hidden_dim))
			layers.append(nn.Tanh())
			last_dim = hidden_dim
		layers.append(nn.Linear(last_dim, output_dim))
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


@dataclass
class Sample:
    obs: np.ndarray
    mask: np.ndarray
    target: np.ndarray
    weight: float = 1.0

class ReservoirBuffer:
	def __init__(self, capacity: int, rng: np.random.Generator | None = None):
		self.capacity = capacity
		self.data: list[Sample] = []
		self.seen = 0
		self.rng = rng if rng is not None else np.random.default_rng()

	def __len__(self) -> int:
		return len(self.data)

	def add(self, obs: np.ndarray, mask: np.ndarray, target: np.ndarray, weight: float = 1.0) -> None:
		sample = Sample(obs=obs.astype(np.float32), mask=mask.astype(np.bool_), target=target.astype(np.float32), weight=float(weight))
		self.seen += 1
		if len(self.data) < self.capacity:
			self.data.append(sample)
			return
		idx = int(self.rng.integers(0, self.seen))
		if idx < self.capacity:
			self.data[idx] = sample

	def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
		if not self.data:
			raise ValueError("Cannot sample from an empty buffer")
		size = min(batch_size, len(self.data))
		indices = self.rng.choice(len(self.data), size=size, replace=False)
		batch = [self.data[int(i)] for i in indices]
		obs = torch.as_tensor(np.stack([item.obs for item in batch]), dtype=torch.float32, device=device)
		mask = torch.as_tensor(np.stack([item.mask for item in batch]), dtype=torch.bool, device=device)
		target = torch.as_tensor(np.stack([item.target for item in batch]), dtype=torch.float32, device=device)
		weight = torch.as_tensor(np.asarray([item.weight for item in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
		return {"obs": obs, "mask": mask, "target": target, "weight": weight}


def masked_probs_from_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	masked_logits = logits.masked_fill(~mask, -1e9)
	probs = torch.softmax(masked_logits, dim=-1)
	probs = probs * mask.float()
	denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
	return probs / denom


def regret_matching(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	positive = torch.clamp(advantages, min=0.0) * mask.float()
	normalizer = positive.sum(dim=-1, keepdim=True)
	legal = mask.float()
	legal_count = legal.sum(dim=-1, keepdim=True).clamp_min(1.0)
	uniform = legal / legal_count
	probs = torch.where(normalizer > 0, positive / normalizer.clamp_min(1e-8), uniform)
	probs = probs * legal
	probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
	return probs


def legal_action_indices(mask: np.ndarray) -> np.ndarray:
	return np.flatnonzero(mask.astype(bool))


def sample_action(probs: torch.Tensor, rng: np.random.Generator) -> int:
	probs_np = probs.detach().cpu().numpy().astype(np.float64)
	probs_np = probs_np / probs_np.sum()
	return int(rng.choice(len(probs_np), p=probs_np))


def policy_distribution(
	net: nn.Module,
	obs: np.ndarray,
	mask: np.ndarray,
	device: torch.device,
	use_regret_matching: bool = False,
) -> torch.Tensor:
	obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
	mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
	logits = net(obs_t)
	if use_regret_matching:
		probs = regret_matching(logits, mask_t)
	else:
		probs = masked_probs_from_logits(logits, mask_t)
	return probs.squeeze(0)


def current_strategy(
	traverser_idx: int,
	player_idx: int,
	obs: np.ndarray,
	mask: np.ndarray,
	advantage_net: nn.Module,
	average_policy_net: nn.Module,
	device: torch.device,
) -> torch.Tensor:
	return policy_distribution(advantage_net, obs, mask, device, use_regret_matching=True)



def rollout_to_terminal(
	env: SushiGoParallelEnv,
	observations: dict[str, dict[str, np.ndarray]],
	target_agent: str,
	traverser_idx: int,
	advantage_net: nn.Module,
	average_policy_net: nn.Module,
	device: torch.device,
	rng: np.random.Generator,
	fixed_action: int | None = None,
) -> float:
	sim_env = copy.deepcopy(env)
	total_return = 0.0
	next_obs = observations
	first_step = True

	while sim_env.agents:
		active_agents = list(sim_env.agents)
		actions: dict[str, int] = {}
		for player_idx, agent in enumerate(active_agents):
			obs = next_obs[agent]["observation"]
			mask = next_obs[agent]["action_mask"]
			if first_step and agent == target_agent and fixed_action is not None:
				actions[agent] = int(fixed_action)
				continue
			probs = current_strategy(
				traverser_idx=traverser_idx,
				player_idx=player_idx,
				obs=obs,
				mask=mask,
				advantage_net=advantage_net,
				average_policy_net=average_policy_net,
				device=device,
			)
			actions[agent] = sample_action(probs, rng)

		next_obs, rewards, _, _, _ = sim_env.step(actions)
		total_return += float(rewards.get(target_agent, 0.0))
		first_step = False

	return total_return


def estimate_advantages_for_state(
	env: SushiGoParallelEnv,
	observations: dict[str, dict[str, np.ndarray]],
	traverser_idx: int,
	advantage_net: nn.Module,
	average_policy_net: nn.Module,
	device: torch.device,
	rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	target_agent = env.possible_agents[traverser_idx]
	obs = observations[target_agent]["observation"]
	mask = observations[target_agent]["action_mask"].astype(bool)
	legal_actions = legal_action_indices(mask)
	if legal_actions.size == 0:
		raise RuntimeError("The environment produced a state with no legal actions")

	action_values = []
	for action in legal_actions:
		value = rollout_to_terminal(
			env=env,
			observations=observations,
			target_agent=target_agent,
			traverser_idx=traverser_idx,
			advantage_net=advantage_net,
			average_policy_net=average_policy_net,
			device=device,
			rng=rng,
			fixed_action=int(action),
		)
		action_values.append(value)

	action_values_arr = np.asarray(action_values, dtype=np.float32)
	policy_probs = policy_distribution(
        advantage_net,
        obs,
        mask,
        device=device,
        use_regret_matching=True,
    ).detach().cpu().numpy().astype(np.float32)

	baseline = float(np.sum(policy_probs[legal_actions] * action_values_arr))
	advantages = np.zeros(N_TYPES, dtype=np.float32)
	for action, value in zip(legal_actions, action_values_arr):
		advantages[int(action)] = float(value - baseline)

	return obs, mask, advantages


def optimize_advantage_net(
	net: nn.Module,
	optimizer: torch.optim.Optimizer,
	buffer: ReservoirBuffer,
	batch_size: int,
	updates: int,
	device: torch.device,
) -> float:
	if len(buffer) == 0:
		return 0.0

	last_loss = 0.0
	for _ in range(updates):
		batch = buffer.sample(batch_size, device=device)
		prediction = net(batch["obs"])
		mask = batch["mask"].float()
		target = batch["target"]
		loss = (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
		optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(net.parameters(), 10.0)
		optimizer.step()
		last_loss = float(loss.item())
	return last_loss


def optimize_average_policy_net(
	net: nn.Module,
	optimizer: torch.optim.Optimizer,
	buffer: ReservoirBuffer,
	batch_size: int,
	updates: int,
	device: torch.device,
) -> float:
	if len(buffer) == 0:
		return 0.0

	last_loss = 0.0
	for _ in range(updates):
		batch = buffer.sample(batch_size, device=device)
		logits = net(batch["obs"])
		log_probs = torch.log(masked_probs_from_logits(logits, batch["mask"]).clamp_min(1e-8))
		target = batch["target"]
		mask = batch["mask"].float()
		per_sample_loss = -((target * log_probs) * mask).sum(dim=-1,keepdim=True,)
		weights = batch["weight"]
		weights = weights / weights.mean().clamp_min(1e-8)
		loss = (per_sample_loss * weights).mean()
		optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(net.parameters(), 10.0)
		optimizer.step()
		last_loss = float(loss.item())
	return last_loss


def play_traversal(
	env: SushiGoParallelEnv,
	traverser_idx: int,
	iteration: int,
	advantage_net: nn.Module,
	average_policy_net: nn.Module,
	advantage_buffer: ReservoirBuffer,
	policy_buffer: ReservoirBuffer,
	device: torch.device,
	rng: np.random.Generator,
) -> dict[str, float]:
	observations, _ = env.reset()
	traverser_agent = env.possible_agents[traverser_idx]
	traversal_reward = 0.0
	recorded_states = 0

	while env.agents:
		active_agents = list(env.agents)
		actions: dict[str, int] = {}

		for player_idx, agent in enumerate(active_agents):
			obs = observations[agent]["observation"]
			mask = observations[agent]["action_mask"].astype(bool)
			probs = current_strategy(
				traverser_idx=traverser_idx,
				player_idx=player_idx,
				obs=obs,
				mask=mask,
				advantage_net=advantage_net,
				average_policy_net=average_policy_net,
				device=device,
			)

			policy_weight = float(iteration + 1) #float(np.sqrt(iteration+1))

			policy_buffer.add(
				obs=obs,
				mask=mask,
				target=probs.detach().cpu().numpy().astype(np.float32),
				weight=policy_weight,
			)

			if agent == traverser_agent:
				obs_t, mask_t, advantages = estimate_advantages_for_state(
					env=env,
					observations=observations,
					traverser_idx=traverser_idx,
					advantage_net=advantage_net,
					average_policy_net=average_policy_net,
					device=device,
					rng=rng,
				)
				advantage_buffer.add(obs_t, mask_t, advantages)
				recorded_states += 1

			actions[agent] = sample_action(probs, rng)

		observations, rewards, _, _, _ = env.step(actions)
		traversal_reward += float(rewards.get(traverser_agent, 0.0))

	return {
		"traversal_reward": traversal_reward,
		"recorded_states": float(recorded_states),
	}


def build_env(args) -> SushiGoParallelEnv:
	return SushiGoParallelEnv(
		n_players=args.n_players,
		history_len=args.history_len,
		include_opponent_tableaus=args.include_opponent_tableaus,
		reward_scale=args.reward_scale,
		zero_sum_rewards=True,
		render_mode=None,
	)

def evaluate_against_random(
    env: SushiGoParallelEnv,
    policy_net: nn.Module,
    eval_episodes: int,
    device: torch.device,
    rng: np.random.Generator,
    greedy: bool = False,
) -> dict[str, float]:
    """Evaluate the average policy network against random legal-action opponents."""
    target_agent = env.possible_agents[0]

    wins = 0
    ties = 0
    total_reward = 0.0

    policy_net.eval()

    with torch.no_grad():
        for _ in range(eval_episodes):
            observations, _ = env.reset()
            episode_reward = 0.0

            while env.agents:
                actions: dict[str, int] = {}

                for agent in list(env.agents):
                    obs = observations[agent]["observation"]
                    mask = observations[agent]["action_mask"].astype(bool)

                    if agent == target_agent:
                        probs = policy_distribution(
                            policy_net,
                            obs,
                            mask,
                            device,
                            use_regret_matching=False,
                        )

                        if greedy:
                            legal = torch.as_tensor(mask, dtype=torch.bool, device=device)
                            masked_probs = probs.masked_fill(~legal, -1.0)
                            actions[agent] = int(torch.argmax(masked_probs).item())
                        else:
                            actions[agent] = sample_action(probs, rng)

                    else:
                        legal = legal_action_indices(mask)
                        actions[agent] = int(rng.choice(legal))

                observations, rewards, _, _, _ = env.step(actions)
                episode_reward += float(rewards.get(target_agent, 0.0))

            if episode_reward > 0:
                wins += 1
            elif episode_reward == 0:
                ties += 1

            total_reward += episode_reward

    policy_net.train()

    return {
        "win_rate": wins / eval_episodes,
        "tie_rate": ties / eval_episodes,
        "mean_reward": total_reward / eval_episodes,
    }

def train(args):
	device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
	set_seed(args.seed)
	if args.n_players != 2:
		raise ValueError("Deep CFR zero-sum training should use --n-players 2")


	if args.smoke:
		args.iterations = 3
		args.traversals_per_iteration = 1
		args.advantage_updates = 2
		args.policy_updates = 2
		args.advantage_buffer_size = 512
		args.policy_buffer_size = 512
		args.batch_size = 64
		args.hidden_dim = 64
		args.depth = 2

	env = build_env(args)
	eval_env = build_env(args)
	obs_dim = env.observation_spaces[env.possible_agents[0]]["observation"].shape[0]

	advantage_net = MLP(obs_dim, N_TYPES, hidden_dim=args.hidden_dim, depth=args.depth).to(device)
	average_policy_net = MLP(obs_dim, N_TYPES, hidden_dim=args.hidden_dim, depth=args.depth).to(device)

	advantage_optimizer = torch.optim.Adam(advantage_net.parameters(), lr=args.lr)
	policy_optimizer = torch.optim.Adam(average_policy_net.parameters(), lr=args.lr)

	advantage_buffer = ReservoirBuffer(args.advantage_buffer_size)
	policy_buffer = ReservoirBuffer(args.policy_buffer_size)
	rng = np.random.default_rng(args.seed)
	os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
	log_file = open(args.log_file, "w", encoding="utf-8")

	print(
		f"device={device}  n_players={args.n_players}  obs_dim={obs_dim}  "
		f"iterations={args.iterations}"
	)

	for iteration in range(args.iterations):
		traversal_rewards = []
		recorded_states = []

		for traversal_idx in range(args.traversals_per_iteration):
			traverser_idx = (iteration * args.traversals_per_iteration + traversal_idx) % args.n_players
			stats = play_traversal(
				env=env,
				traverser_idx=traverser_idx,
				iteration=iteration,
				advantage_net=advantage_net,
				average_policy_net=average_policy_net,
				advantage_buffer=advantage_buffer,
				policy_buffer=policy_buffer,
				device=device,
				rng=rng,
			)
			traversal_rewards.append(stats["traversal_reward"])
			recorded_states.append(stats["recorded_states"])

		advantage_loss = optimize_advantage_net(
			net=advantage_net,
			optimizer=advantage_optimizer,
			buffer=advantage_buffer,
			batch_size=args.batch_size,
			updates=args.advantage_updates,
			device=device,
		)
		policy_loss = optimize_average_policy_net(
			net=average_policy_net,
			optimizer=policy_optimizer,
			buffer=policy_buffer,
			batch_size=args.batch_size,
			updates=args.policy_updates,
			device=device,
		)

		
		eval_stats = None
		if args.eval_every > 0 and (
			iteration == 0 or (iteration + 1) % args.eval_every == 0
		):
			eval_stats = evaluate_against_random(
				env=eval_env,
				policy_net=average_policy_net,
				eval_episodes=args.eval_episodes,
				device=device,
				rng=rng,
				greedy=args.eval_greedy,
			)

		msg = (
			f"iter {iteration:3d} | traversals={len(traversal_rewards):2d} | "
			f"adv_loss={advantage_loss:.4f} | policy_loss={policy_loss:.4f} | "
			f"mean_traversal_reward={float(np.mean(traversal_rewards)):+.3f} | "
			f"adv_buffer={len(advantage_buffer):5d} | policy_buffer={len(policy_buffer):5d}"
		)

		if eval_stats is not None:
			msg += (
				f" | eval_random_win={eval_stats['win_rate']:.3f}"
				f" | eval_random_tie={eval_stats['tie_rate']:.3f}"
				f" | eval_random_reward={eval_stats['mean_reward']:+.3f}"
			)

		print(msg)
		with open("deepcfr_training_log.txt", "a", encoding="utf-8") as f:
			f.write(msg + "\n")

	os.makedirs(args.save_dir, exist_ok=True)
	advantage_path = os.path.join(args.save_dir, f"sushi_go_deep_cfr_advantage_{args.n_players}_players.pt")
	policy_path = os.path.join(args.save_dir, f"sushi_go_deep_cfr_policy_{args.n_players}_players.pt")

	torch.save(
		{
			"model_state_dict": advantage_net.state_dict(),
			"obs_dim": obs_dim,
			"n_players": args.n_players,
			"hidden_dim": args.hidden_dim,
			"depth": args.depth,
		},
		advantage_path,
	)
	torch.save(
		{
			"model_state_dict": average_policy_net.state_dict(),
			"obs_dim": obs_dim,
			"n_players": args.n_players,
			"hidden_dim": args.hidden_dim,
			"depth": args.depth,
		},
		policy_path,
	)
	print(f"saved advantage net -> {advantage_path}")
	print(f"saved average policy -> {policy_path}")


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--n-players", type=int, default=2, choices=[2, 3, 4])
	parser.add_argument("--iterations", type=int, default=1000)
	parser.add_argument("--traversals-per-iteration", type=int, default=5)
	parser.add_argument("--advantage-updates", type=int, default=10)
	parser.add_argument("--policy-updates", type=int, default=10)
	parser.add_argument("--advantage-buffer-size", type=int, default=100000)
	parser.add_argument("--policy-buffer-size", type=int, default=100000)
	parser.add_argument("--batch-size", type=int, default=512)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--hidden-dim", type=int, default=256)
	parser.add_argument("--depth", type=int, default=3)
	parser.add_argument("--reward-scale", type=float, default=0.1)
	parser.add_argument("--history-len", type=int, default=None)
	parser.add_argument("--include-opponent-tableaus", action="store_true", default=True)
	parser.add_argument("--no-include-opponent-tableaus", dest="include_opponent_tableaus", action="store_false")
	parser.add_argument("--cuda", action="store_true")
	parser.add_argument("--seed", type=int, default=7)
	parser.add_argument("--smoke", action="store_true", help="tiny wiring-check run")
	parser.add_argument("--save-dir", type=str, default=os.path.join("models", "DeepCFR"))
	parser.add_argument("--eval-every", type=int, default=1)
	parser.add_argument("--eval-episodes", type=int, default=100)
	parser.add_argument(
    "--eval-greedy",
    action="store_true",
    help="Evaluate DeepCFR with greedy argmax instead of sampling from average policy",
	),
	parser.add_argument(
    "--log-file",
    type=str,
    default=os.path.join("models", "DeepCFR", "training_log.txt"),
    help="Path to save training logs as txt",
	)
	return parser.parse_args()


if __name__ == "__main__":
	train(get_args())
