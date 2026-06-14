"""Launch reproducible batches of isolated league training runs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "src" / "train" / "train_dqn.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "league"


@dataclass(frozen=True)
class LeaguePreset:
    name: str
    player_args: tuple[tuple[str, int], ...]

    def training_args(self) -> dict[str, object]:
        """Return the player configuration plus shared league hyperparameters."""
        return {
            **dict(self.player_args),
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


PRESETS = {
    preset.name: preset
    for preset in (
        LeaguePreset("fixed_2p", (("n_players", 2),)),
        LeaguePreset("fixed_3p", (("n_players", 3),)),
        LeaguePreset("fixed_4p", (("n_players", 4),)),
        LeaguePreset(
            "variable_2_4",
            (("min_n_players", 2), ("max_n_players", 4)),
        ),
    )
}


@dataclass(frozen=True)
class RunSpec:
    preset: LeaguePreset
    repetition: int
    run_dir: Path
    use_cuda: bool

    @property
    def label(self) -> str:
        return f"{self.preset.name}/repetition_{self.repetition}"

    @property
    def model_path(self) -> Path:
        return self.run_dir / "model.pt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.csv"

    @property
    def stdout_path(self) -> Path:
        return self.run_dir / "stdout.log"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def complete_path(self) -> Path:
        return self.run_dir / "COMPLETE"

    def command(self, python_executable: str = sys.executable) -> list[str]:
        args = self.preset.training_args()
        command = [python_executable, str(TRAIN_SCRIPT)]
        for key, value in args.items():
            command.extend((f"--{key.replace('_', '-')}", str(value)))
        command.extend(("--save-path", str(self.model_path)))
        command.extend(("--log-path", str(self.metrics_path)))
        if self.use_cuda:
            command.append("--cuda")
        return command


def build_run_specs(
    preset_names: Sequence[str],
    repetitions: int,
    output_root: Path,
    use_cuda: bool,
) -> list[RunSpec]:
    return [
        RunSpec(
            preset=PRESETS[preset_name],
            repetition=repetition,
            run_dir=output_root / preset_name / f"repetition_{repetition}",
            use_cuda=use_cuda,
        )
        for preset_name in preset_names
        for repetition in range(1, repetitions + 1)
    ]


def preflight(specs: Sequence[RunSpec]) -> tuple[list[RunSpec], list[RunSpec]]:
    """Split runs into pending/completed and reject incomplete directories."""
    pending: list[RunSpec] = []
    completed: list[RunSpec] = []
    incomplete: list[Path] = []
    for spec in specs:
        if not spec.run_dir.exists():
            pending.append(spec)
        elif spec.complete_path.is_file():
            completed.append(spec)
        else:
            incomplete.append(spec.run_dir)
    if incomplete:
        paths = "\n".join(f"  - {path}" for path in incomplete)
        raise RuntimeError(
            "Incomplete run directories already exist. Delete them before rerunning:\n"
            f"{paths}"
        )
    return pending, completed


def _git_metadata() -> dict[str, object]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "torchrl", "tensordict", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run_metadata(spec: RunSpec, command: Sequence[str]) -> dict[str, object]:
    return {
        "preset": spec.preset.name,
        "repetition": spec.repetition,
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "training_args": spec.preset.training_args(),
        "cuda_requested": spec.use_cuda,
        "command": list(command),
        "git": _git_metadata(),
        "python": sys.version,
        "packages": _package_versions(),
    }


def run_training(spec: RunSpec) -> bool:
    """Run one subprocess, stream its output, and mark validated success."""
    spec.run_dir.mkdir(parents=True, exist_ok=False)
    command = spec.command()
    spec.config_path.write_text(
        json.dumps(run_metadata(spec, command), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with spec.stdout_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            print(f"[{spec.label}] {line}", end="", flush=True)
        return_code = process.wait()

    artifacts_ok = spec.model_path.is_file() and spec.metrics_path.is_file()
    if return_code == 0 and artifacts_ok:
        spec.complete_path.write_text(
            datetime.now(timezone.utc).isoformat() + "\n",
            encoding="utf-8",
        )
        return True

    print(
        f"[{spec.label}] FAILED: exit={return_code}, "
        f"model={spec.model_path.is_file()}, metrics={spec.metrics_path.is_file()}",
        file=sys.stderr,
    )
    return False


def run_pending(
    specs: Sequence[RunSpec],
    parallelism: int,
    runner: Callable[[RunSpec], bool] = run_training,
) -> bool:
    """Run all jobs with bounded concurrency and report aggregate success."""
    succeeded = True
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {executor.submit(runner, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                run_ok = future.result()
            except Exception as error:  # noqa: BLE001 - isolate failed runs
                print(f"[{spec.label}] FAILED: {error}", file=sys.stderr)
                run_ok = False
            succeeded = run_ok and succeeded
    return succeeded


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=tuple(PRESETS),
        default=list(PRESETS),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.parallelism < 1:
        parser.error("--parallelism must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = build_run_specs(args.presets, args.repetitions, args.output_root, args.cuda)
    try:
        pending, completed = preflight(specs)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2

    for spec in completed:
        print(f"SKIP {spec.label}: complete")
    for spec in pending:
        print(f"RUN  {spec.label}: {' '.join(spec.command())}")

    if args.dry_run or not pending:
        return 0
    return 0 if run_pending(pending, args.parallelism) else 1


if __name__ == "__main__":
    raise SystemExit(main())
