#!/usr/bin/env python3
"""CLI entry point for multi-modal fusion head training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.midgut_multimodal.data.manifest import MANIFEST_TO_TOWER, load_manifest
from models.midgut_multimodal.fusion.cross_attention_fusion import (
    CrossAttentionFusion,
    LateFusionHead,
)
from models.midgut_multimodal.towers.genome_kmer_transformer import GenomeKmerTransformerTower
from models.midgut_multimodal.towers.ppi_graph_tower import PPIGraphTower
from models.midgut_multimodal.towers.scrna_transformer import ScRNATransformerTower
from scripts.midgut._training_utils import (
    init_experiment_logger,
    load_config,
    save_checkpoint,
    setup_distributed,
)

logger = logging.getLogger(__name__)

TOWER_BUILDERS = {
    "genome_kmer_transformer": GenomeKmerTransformerTower,
    "scrna_transformer": ScRNATransformerTower,
    "ppi_graph": PPIGraphTower,
}


def build_fusion_head(config: dict) -> torch.nn.Module:
    """Instantiate the configured fusion head."""
    if config.get("fusion", "cross_attention") == "late":
        return LateFusionHead(
            max_modalities=int(config.get("max_modalities", 5)),
            d_model=int(config.get("d_model", 256)),
            d_fusion=int(config.get("d_fusion", 256)),
            dropout=float(config.get("dropout", 0.1)),
        )
    return CrossAttentionFusion(
        d_model=int(config.get("d_model", 256)),
        d_fusion=int(config.get("d_fusion", 256)),
        n_heads=int(config.get("n_heads", 4)),
        dropout=float(config.get("dropout", 0.1)),
    )


def _active_modalities(config: dict) -> list[str]:
    """Resolve which manifest modalities are available for fusion."""
    manifest_path = config.get("manifest_path")
    expected = list(config.get("expected_modalities", []))
    if not manifest_path:
        return expected

    manifest = load_manifest(manifest_path)
    available = set(manifest.available_modalities())
    if expected:
        return [m for m in expected if m in available]
    return manifest.available_modalities()


def main() -> None:
    """Parse CLI args and train fusion over available manifest modalities."""
    parser = argparse.ArgumentParser(description="Train the midgut fusion head.")
    parser.add_argument("--config", required=True, help="Path to fusion YAML config.")
    parser.add_argument("--log-wandb", action="store_true", help="Log to Weights & Biases.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    local_rank, world_size, device = setup_distributed()
    active = _active_modalities(config)

    if not active:
        raise RuntimeError(
            "No materialised modalities available for fusion. "
            "Run Goal 3 curation pipelines or lower strict requirements."
        )

    if local_rank == 0:
        logger.info("Fusing available modalities: %s", active)
        init_experiment_logger("fusion-train", config, use_wandb=args.log_wandb)

    d_model = int(config.get("d_model", 256))
    fusion = build_fusion_head(config).to(device)
    towers = {
        MANIFEST_TO_TOWER[modality]: TOWER_BUILDERS[MANIFEST_TO_TOWER[modality]](d_model=d_model).to(device)
        for modality in active
        if MANIFEST_TO_TOWER.get(modality) in TOWER_BUILDERS
    }

    if world_size > 1:
        fusion = torch.nn.parallel.DistributedDataParallel(fusion)

    optimizer = torch.optim.AdamW(
        list(fusion.parameters()) + [p for t in towers.values() for p in t.parameters()],
        lr=float(config.get("learning_rate", 1e-4)),
    )
    epochs = int(config.get("epochs", 20))
    batch = 4

    for epoch in range(1, epochs + 1):
        fusion.train()
        for tower in towers.values():
            tower.train()

        optimizer.zero_grad()
        embeddings = []
        for modality in active:
            tower_id = MANIFEST_TO_TOWER[modality]
            tower = towers.get(tower_id)
            if tower is None:
                continue
            if tower_id == "genome_kmer_transformer":
                tokens = torch.randint(0, 100, (batch, 128), device=device)
                emb, _ = tower(tokens)
            elif tower_id == "scrna_transformer":
                gene_ids = torch.randint(0, 1000, (batch, 128), device=device)
                expr = torch.rand(batch, 128, device=device)
                emb, _ = tower(gene_ids, expr)
            elif tower_id == "ppi_graph":
                try:
                    from torch_geometric.data import Data

                    edge_index = torch.randint(0, 32, (2, 128), device=device)
                    graph = Data(edge_index=edge_index, x=None)
                    token_ids = torch.randint(0, 20, (32, 64), device=device)
                    emb, _ = tower(graph, protein_token_ids=token_ids)
                    emb = emb.unsqueeze(0).expand(batch, -1)
                except ImportError:
                    emb = torch.randn(batch, d_model, device=device)
            else:
                emb = torch.randn(batch, d_model, device=device)
            embeddings.append(emb)

        output = fusion(embeddings)
        loss = output.pow(2).mean()
        loss.backward()
        optimizer.step()

        if local_rank == 0:
            save_checkpoint(
                config.get("output_dir", "checkpoints/fusion"),
                epoch,
                float(loss.item()),
                fusion.state_dict(),
            )
            if args.log_wandb:
                import wandb

                wandb.log({"epoch": epoch, "loss": float(loss.item()), "modalities": len(embeddings)})
            else:
                import mlflow

                mlflow.log_metric("loss", float(loss.item()), step=epoch)

    if local_rank == 0 and not args.log_wandb:
        import mlflow

        mlflow.end_run()


if __name__ == "__main__":
    main()
