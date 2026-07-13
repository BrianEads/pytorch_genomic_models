"""Training utilities shared by midgut CLI scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML training configuration file."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialise DDP rank/world-size and return the local device."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return local_rank, world_size, device


def init_experiment_logger(
    run_name: str,
    config: dict[str, Any],
    use_wandb: bool = False,
) -> Any:
    """Start MLflow or W&B experiment tracking."""
    if use_wandb:
        import wandb

        wandb.init(project="midgut-multimodal", name=run_name, config=config)
        return wandb

    import mlflow

    mlflow.set_experiment("midgut-multimodal")
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({k: str(v) for k, v in config.items() if not isinstance(v, dict)})
    return mlflow


def save_checkpoint(
    output_dir: str,
    epoch: int,
    loss: float,
    state_dict: dict[str, Any],
) -> Path:
    """Save a checkpoint with the canonical filename format."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = path / f"epoch={epoch}_loss={loss:.4f}.pt"
    torch.save(state_dict, checkpoint_path)
    latest = path / "latest.pt"
    torch.save(state_dict, latest)
    return checkpoint_path
