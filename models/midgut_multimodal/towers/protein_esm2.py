"""ESM-2 protein sequence tower with optional LoRA fine-tuning."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ProteinESM2Tower(nn.Module):
    """ESM-2 wrapper that mean-pools residue embeddings to a CLS vector.

    When ``use_stub=True`` (default for unit tests), a lightweight embedding
    encoder is used instead of loading ``fair-esm`` weights.

    Args:
        d_model: Output embedding dimension after projection.
        max_length: Maximum amino-acid sequence length.
        esm_model_name: HuggingFace / fair-esm checkpoint name.
        use_lora: Apply LoRA adapters when a real ESM-2 backbone is loaded.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        use_stub: If True, skip heavyweight ESM-2 weight download.

    Input shape:
        token_ids: ``(B, L)`` amino-acid token indices in ``[0, vocab_size)``.

    Output:
        Tuple of mean-pooled embedding ``(B, d_model)`` and ``None``.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_length: int = 1024,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        use_stub: bool = True,
        vocab_size: int = 33,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.use_stub = use_stub
        self._esm_backbone: Optional[nn.Module] = None

        if use_stub:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=d_model * 4,
                    batch_first=True,
                ),
                num_layers=2,
            )
        else:
            self._init_esm_backbone(
                esm_model_name=esm_model_name,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
            )

        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

    def _init_esm_backbone(
        self,
        esm_model_name: str,
        use_lora: bool,
        lora_r: int,
        lora_alpha: int,
    ) -> None:
        """Load ESM-2 and optionally wrap with LoRA adapters."""
        try:
            import esm  # type: ignore[import-untyped]
            from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "fair-esm and peft are required when use_stub=False. "
                "Install with: pip install fair-esm peft"
            ) from exc

        model, _ = esm.pretrained.load_model_and_alphabet(esm_model_name)
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
            )
            model = get_peft_model(model, lora_config)
        self._esm_backbone = model

    def encode_nodes(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Encode protein sequences into per-node embeddings for PPI graphs.

        Used when Goal 3 PPI graphs ship with ``x=None``; Goal 2 owns filling
        node features before graph convolution.

        Args:
            token_ids: Amino-acid token indices, shape ``(num_nodes, L)``.

        Returns:
            Projected node embeddings ``(num_nodes, d_model)`` and ``None``.
        """
        embedding, _ = self.forward(token_ids)
        return embedding, None

    def fill_graph_node_features(
        self,
        graph: object,
        protein_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Write ESM-2 embeddings into a PyG graph's ``x`` attribute in-place.

        Args:
            graph: PyG ``Data`` object whose ``x`` may be ``None``.
            protein_token_ids: Tokenised sequences aligned to graph nodes.

        Returns:
            The populated node feature tensor ``(num_nodes, d_model)``.
        """
        node_features, _ = self.encode_nodes(protein_token_ids)
        if hasattr(graph, "x"):
            graph.x = node_features
        return node_features

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode protein sequences.

        Args:
            token_ids: Amino-acid token indices, shape ``(B, L)``.

        Returns:
            Mean-pooled sequence embedding and ``None`` for attention weights.
        """
        if self.use_stub:
            tokens = self.token_embedding(token_ids)
            encoded = self.encoder(tokens)
            pooled = encoded.mean(dim=1)
        else:
            assert self._esm_backbone is not None
            results = self._esm_backbone(token_ids, repr_layers=[self._esm_backbone.num_layers])
            residue_repr = results["representations"][self._esm_backbone.num_layers]
            pooled = residue_repr.mean(dim=1)

        embedding = self.projection(pooled)
        return embedding, None
