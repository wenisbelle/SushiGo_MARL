"""Checkpoint discovery and greedy DQN inference for league play."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import torch
from tensordict import TensorDict


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from SushiGo_env.sushi_go_env import OBS_COMPONENTS
from SushiGo_env.torchrl_integration import (
    ACTION_KEY,
    GROUP,
    MASK_KEY,
    PLAYER_MASK_KEY,
    flat_observation_dim,
    make_torchrl_env,
)
from train.train_dqn import build_arg_parser, build_qvalue_actor, resolve_player_config


ELIGIBLE_2P_PRESETS = (
    "fixed_2p",
    "variable_2_4",
    "variable_encoder_2_4",
)
ELIGIBLE_PRESETS_BY_PLAYERS = {
    2: ELIGIBLE_2P_PRESETS,
    3: ("fixed_3p", "variable_2_4", "variable_encoder_2_4"),
    4: ("fixed_4p", "variable_2_4", "variable_encoder_2_4"),
}


@dataclass(frozen=True)
class CheckpointSpec:
    competitor: str
    repetition: int
    run_dir: Path
    checkpoint_path: Path
    training_args: Mapping[str, object]

    @property
    def label(self) -> str:
        return f"{self.competitor}/repetition_{self.repetition}"


def training_defaults() -> dict[str, object]:
    """Return current model defaults without duplicating training definitions."""
    return vars(build_arg_parser().parse_args([]))


def discover_checkpoints(
    models_root: Path,
    competitors: Sequence[str] = ELIGIBLE_2P_PRESETS,
) -> list[CheckpointSpec]:
    """Discover completed, eligible checkpoint repetitions in stable order."""
    requested = set(competitors)
    known = set().union(*ELIGIBLE_PRESETS_BY_PLAYERS.values())
    unknown = requested.difference(known)
    if unknown:
        raise ValueError(f"Unknown two-player competitors: {', '.join(sorted(unknown))}")

    checkpoints: list[CheckpointSpec] = []
    preset_order = {
        name: index
        for index, name in enumerate(dict.fromkeys(
            preset
            for presets in ELIGIBLE_PRESETS_BY_PLAYERS.values()
            for preset in presets
        ))
    }
    for competitor in preset_order:
        if competitor not in requested:
            continue
        preset_dir = models_root / competitor
        if not preset_dir.is_dir():
            continue
        for run_dir in preset_dir.glob("repetition_*"):
            if not (run_dir / "COMPLETE").is_file():
                continue
            config_path = run_dir / "config.json"
            checkpoint_path = run_dir / "model.pt"
            missing = [
                path.name
                for path in (config_path, checkpoint_path)
                if not path.is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"Completed run {run_dir} is missing required artifacts: "
                    f"{', '.join(missing)}"
                )
            try:
                repetition = int(run_dir.name.removeprefix("repetition_"))
            except ValueError as error:
                raise RuntimeError(f"Invalid repetition directory: {run_dir}") from error

            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("preset") != competitor:
                raise RuntimeError(
                    f"Preset mismatch in {config_path}: expected {competitor!r}, "
                    f"got {config.get('preset')!r}"
                )
            if config.get("repetition") not in (None, repetition):
                raise RuntimeError(
                    f"Repetition mismatch in {config_path}: expected {repetition}, "
                    f"got {config.get('repetition')!r}"
                )
            training_args = config.get("training_args")
            if not isinstance(training_args, dict):
                raise RuntimeError(f"Missing training_args object in {config_path}")
            checkpoints.append(
                CheckpointSpec(
                    competitor=competitor,
                    repetition=repetition,
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    training_args=training_args,
                )
            )
    return sorted(
        checkpoints,
        key=lambda spec: (preset_order[spec.competitor], spec.repetition),
    )


def checkpoint_matchups(
    checkpoints: Sequence[CheckpointSpec],
    table_size: int,
) -> list[tuple[CheckpointSpec, ...]]:
    """Return unordered distinct-checkpoint tables, excluding all-same models."""
    return [
        matchup
        for matchup in combinations(checkpoints, table_size)
        if table_size == 2 or len({spec.competitor for spec in matchup}) > 1
    ]


def checkpoint_pairs(checkpoints: Sequence[CheckpointSpec]):
    """Backward-compatible two-player matchup helper."""
    return checkpoint_matchups(checkpoints, 2)


def native_observations(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    model_n_players: int,
) -> list[Mapping[str, np.ndarray]]:
    """Render canonical four-slot state in a checkpoint's native observation shape."""
    rendered = []
    sequence_rows = model_n_players - 1
    for seat in range(model_n_players):
        source = observations[f"player_{seat}"]
        native = dict(source)
        native["hand_history"] = source["hand_history"][:sequence_rows]
        native["opponent_tableaus"] = source["opponent_tableaus"][:sequence_rows]
        rendered.append(native)
    return rendered


def observations_to_tensordict(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    model_n_players: int,
    device: torch.device,
) -> TensorDict:
    """Build the minimal grouped TensorDict consumed by a trained Q actor."""
    native = native_observations(observations, model_n_players)
    td = TensorDict({}, batch_size=[], device=device)
    for component in OBS_COMPONENTS:
        values = np.stack([obs[component] for obs in native])
        td.set(
            (GROUP, "observation", component),
            torch.as_tensor(values, dtype=torch.float32, device=device),
        )
    td.set(
        MASK_KEY,
        torch.as_tensor(
            np.stack([obs["action_mask"] for obs in native]),
            dtype=torch.bool,
            device=device,
        ),
    )
    td.set(
        PLAYER_MASK_KEY,
        torch.as_tensor(
            np.asarray([obs["player_mask"] for obs in native])[:, None],
            dtype=torch.bool,
            device=device,
        ),
    )
    return td


class LoadedPolicy:
    """A validated checkpoint plus its native observation/inference contract."""

    def __init__(self, spec: CheckpointSpec, actor, model_n_players: int, device):
        self.spec = spec
        self.actor = actor
        self.model_n_players = model_n_players
        self.device = torch.device(device)

    def action(self, observations, seat: int) -> int:
        td = observations_to_tensordict(
            observations, self.model_n_players, self.device
        )
        with torch.inference_mode():
            self.actor(td)
        return int(td.get(ACTION_KEY)[seat].item())


def load_policy(spec: CheckpointSpec, device: str = "cpu") -> LoadedPolicy:
    """Rebuild and strictly validate one actor from its recorded configuration."""
    resolved = training_defaults()
    resolved.update(spec.training_args)
    args = Namespace(**resolved)
    n_players, min_n_players, max_n_players, model_n_players = resolve_player_config(args)

    env = make_torchrl_env(
        n_players=n_players,
        min_n_players=min_n_players,
        max_n_players=max_n_players,
        reward_scale=args.reward_scale,
        device=device,
    )
    try:
        actor = build_qvalue_actor(
            env,
            model_n_players,
            flat_observation_dim(env),
            args,
            device=device,
        )
        actor(env.reset())
        state_dict = torch.load(
            spec.checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        actor.load_state_dict(state_dict, strict=True)
        actor.eval()
    except Exception as error:
        raise RuntimeError(f"Could not load checkpoint {spec.label}: {error}") from error
    finally:
        env.close()
    return LoadedPolicy(spec, actor, model_n_players, device)
