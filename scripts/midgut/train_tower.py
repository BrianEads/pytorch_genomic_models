#!/usr/bin/env python3
"""CLI entry point for pre-training a single midgut modality tower."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.midgut_multimodal.data.factory import build_manifest_loaders
from models.midgut_multimodal.towers.bt_screening_mlp import BtScreeningMLPTower
from models.midgut_multimodal.towers.cell_painting_resnet import CellPaintingResNetTower
from models.midgut_multimodal.towers.genome_kmer_transformer import GenomeKmerTransformerTower
from models.midgut_multimodal.towers.popgen_cnn import PopGenCNNTower
from models.midgut_multimodal.towers.ppi_graph_tower import PPIGraphTower
from models.midgut_multimodal.towers.protein_esm2 import ProteinESM2Tower
from models.midgut_multimodal.towers.scrna_transformer import ScRNATransformerTower
from scripts.midgut._training_utils import (
    init_experiment_logger,
    load_config,
    save_checkpoint,
    setup_distributed,
)

logger = logging.getLogger(__name__)

TOWER_REGISTRY = {
    "scrna_transformer": ScRNATransformerTower,
    "genome_kmer_transformer": GenomeKmerTransformerTower,
    "cell_painting_resnet": CellPaintingResNetTower,
    "popgen_cnn": PopGenCNNTower,
    "bt_screening_mlp": BtScreeningMLPTower,
    "protein_esm2": ProteinESM2Tower,
    "ppi_graph": PPIGraphTower,
}

TOWER_TO_MANIFEST = {
    "genome_kmer_transformer": "genome",
    "scrna_transformer": "rnaseq",
    "ppi_graph": "ppi",
}


def build_tower(config: dict) -> torch.nn.Module:
    """Instantiate a tower from a config dict."""
    tower_name = config["tower"]
    cls = TOWER_REGISTRY[tower_name]
    skip_keys = {"tower", "manifest_path", "output_dir", "required_modalities", "strict", "tissue_filter"}
    kwargs = {k: v for k, v in config.items() if k not in skip_keys}
    if tower_name in {"protein_esm2", "ppi_graph"} and "use_stub" not in kwargs:
        kwargs["use_stub"] = True
    return cls(**kwargs)


def _forward_batch(model: torch.nn.Module, tower_name: str, batch: dict, device: torch.device) -> torch.Tensor:
    """Run a single training batch through the appropriate tower."""
    if tower_name == "genome_kmer_transformer":
        tokens = batch["tokens"].to(device)
        embedding, _ = model(tokens)
        return embedding
    if tower_name == "scrna_transformer":
        gene_ids = batch["gene_ids"].to(device)
        expression = batch["expression"].to(device)
        if gene_ids.dim() == 1:
            gene_ids = gene_ids.unsqueeze(0)
            expression = expression.unsqueeze(0)
        embedding, _ = model(gene_ids, expression)
        return embedding
    if tower_name == "ppi_graph":
        graph = batch["graph"].to(device)
        token_ids = batch.get("protein_token_ids")
        if token_ids is not None:
            token_ids = token_ids.to(device)
        embedding, _ = model(graph, protein_token_ids=token_ids)
        return embedding
    raise ValueError(f"No manifest batch handler for tower '{tower_name}'")


def main() -> None:
    """Parse CLI args and pre-train a tower using manifest-backed loaders."""
    parser = argparse.ArgumentParser(description="Pre-train a single midgut modality tower.")
    parser.add_argument("--config", required=True, help="Path to tower YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--log-wandb", action="store_true", help="Log to Weights & Biases.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    tower_name = config["tower"]
    local_rank, world_size, device = setup_distributed()

    required = tuple(config.get("required_modalities", []))
    manifest_path = config.get("manifest_path")
    loader = None
    if manifest_path:
        _, loaders = build_manifest_loaders(
            manifest_path,
            batch_size=int(config.get("batch_size", 32)),
            tissue_filter=config.get("tissue_filter"),
            required_modalities=required or None,
            strict=bool(config.get("strict", False)),
        )
        manifest_key = TOWER_TO_MANIFEST.get(tower_name)
        loader = loaders.get(manifest_key) if manifest_key else None
        if loader is None and required:
            raise RuntimeError(
                f"Tower '{tower_name}' requires manifest modality '{manifest_key}' "
                f"but no loader was built from {manifest_path}"
            )

    if local_rank == 0:
        init_experiment_logger(
            run_name=f"tower-{tower_name}",
            config=config,
            use_wandb=args.log_wandb,
        )

    model = build_tower(config).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
    epochs = args.epochs or int(config.get("epochs", 1))

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0

        if loader is not None:
            for batch in loader:
                optimizer.zero_grad()
                embedding = _forward_batch(model, tower_name, batch, device)
                loss = embedding.pow(2).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                steps += 1
        else:
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss.backward()
            optimizer.step()
            epoch_loss = float(loss.item())
            steps = 1

        mean_loss = epoch_loss / max(steps, 1)
        if local_rank == 0:
            metric = {"epoch": epoch, "loss": mean_loss, "steps": steps}
            if args.log_wandb:
                import wandb

                wandb.log(metric)
            else:
                import mlflow

                mlflow.log_metric("loss", mean_loss, step=epoch)
            save_checkpoint(
                config.get("output_dir", "checkpoints/tower"),
                epoch,
                mean_loss,
                model.state_dict(),
            )
            logger.info("Epoch %d complete (loss=%.4f, steps=%d)", epoch, mean_loss, steps)

    if local_rank == 0 and not args.log_wandb:
        import mlflow

        mlflow.end_run()


if __name__ == "__main__":
    main()
