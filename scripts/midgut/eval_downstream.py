#!/usr/bin/env python3
"""CLI entry point for evaluating downstream midgut tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.midgut_multimodal.downstream.cry_candidate_scorer import CryCandidateScorerHead
from models.midgut_multimodal.downstream.ic50_regression import IC50RegressionHead
from models.midgut_multimodal.downstream.perturbation_predictor import PerturbationPredictorHead
from models.midgut_multimodal.downstream.resistance_classifier import ResistanceClassifierHead
from scripts.midgut._training_utils import init_experiment_logger, load_config, setup_distributed


def evaluate_tasks(config: dict, device: torch.device) -> dict[str, float]:
    """Run stub forward passes for all enabled downstream heads."""
    d_fusion = int(config.get("d_fusion", 256))
    batch = torch.randn(8, d_fusion, device=device)
    metrics: dict[str, float] = {}
    tasks = config.get("tasks", {})

    if tasks.get("ic50_regression", {}).get("enabled", True):
        head = IC50RegressionHead(d_fusion).to(device)
        preds = head(batch)
        targets = torch.randn_like(preds)
        loss = IC50RegressionHead.loss_fn(preds, targets)
        metrics["ic50_loss"] = float(loss.item())

    if tasks.get("resistance_classifier", {}).get("enabled", True):
        head = ResistanceClassifierHead(d_fusion).to(device)
        logits = head(batch)
        targets = torch.randint(0, 2, logits.shape, device=device).float()
        loss = head.loss_fn(logits, targets)
        metrics["resistance_loss"] = float(loss.item())

    if tasks.get("cry_candidate_scorer", {}).get("enabled", True):
        n_labels = int(tasks.get("cry_candidate_scorer", {}).get("n_labels", 4))
        head = CryCandidateScorerHead(d_fusion, n_labels=n_labels).to(device)
        logits = head(batch)
        targets = torch.randint(0, 2, logits.shape, device=device).float()
        loss = CryCandidateScorerHead.loss_fn(logits, targets)
        metrics["cry_loss"] = float(loss.item())

    if tasks.get("perturbation_predictor", {}).get("enabled", True):
        n_types = int(tasks.get("perturbation_predictor", {}).get("n_cell_types", 32))
        head = PerturbationPredictorHead(d_fusion, n_cell_types=n_types).to(device)
        log_probs = head(batch)
        target_dist = torch.softmax(torch.randn(8, n_types, device=device), dim=-1)
        loss = PerturbationPredictorHead.loss_fn(log_probs, target_dist)
        metrics["perturbation_loss"] = float(loss.item())

    return metrics


def main() -> None:
    """Parse CLI args and evaluate downstream task heads."""
    parser = argparse.ArgumentParser(description="Evaluate midgut downstream task heads.")
    parser.add_argument("--config", required=True, help="Path to downstream YAML config.")
    parser.add_argument("--log-wandb", action="store_true", help="Log to Weights & Biases.")
    args = parser.parse_args()

    config = load_config(args.config)
    local_rank, _, device = setup_distributed()
    if local_rank == 0:
        logger = init_experiment_logger("downstream-eval", config, use_wandb=args.log_wandb)
        metrics = evaluate_tasks(config, device)
        for name, value in metrics.items():
            if args.log_wandb:
                import wandb

                wandb.log({name: value})
            else:
                import mlflow

                mlflow.log_metric(name, value)
        print("Downstream evaluation metrics:", metrics)
        if not args.log_wandb:
            import mlflow

            mlflow.end_run()


if __name__ == "__main__":
    main()
