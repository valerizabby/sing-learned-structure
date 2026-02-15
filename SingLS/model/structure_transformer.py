import torch
import torch.nn as nn


class StructureModel(nn.Module):
    def __init__(self, transformer, proj):
        super().__init__()
        self.transformer = transformer
        self.proj = proj

    def forward(self, ssm_batch):
        """
        ssm_batch: [B, T, T]
        returns:   struct [T, B, H]
        """
        struct = self.transformer(ssm_batch)      # [B, T, d_model]
        struct = self.proj(struct)                 # [B, T, H]
        return struct.transpose(0, 1)              # [T, B, H]

import torch
import torch.nn as nn


class StructureTransformer(nn.Module):
    """
    Input:
        ssm_batch: [B, T, T]
    Output:
        struct:    [B, T, d_model]
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()

        # --- Encode each SSM row ---
        self.row_conv = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=3,
            padding=1
        )
        self.row_act = nn.ReLU()
        self.row_pool = nn.AdaptiveAvgPool1d(1)

        # --- Positional encoding (learned) ---
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        self.input_norm = nn.LayerNorm(d_model)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, ssm_batch: torch.Tensor) -> torch.Tensor:
        """
        ssm_batch: [B, T, T]
        returns:   [B, T, d_model]
        """
        B, T, _ = ssm_batch.shape

        if T > self.pos_emb.shape[1]:
            raise ValueError(
                f"SSM length T={T} exceeds max_len={self.pos_emb.shape[1]}"
            )

        # --- Row-wise encoding ---
        rows = ssm_batch.reshape(B * T, 1, T)     # [B*T, 1, T]
        x = self.row_conv(rows)                   # [B*T, d_model, T]
        x = self.row_act(x)
        x = self.row_pool(x).squeeze(-1)          # [B*T, d_model]

        tokens = x.reshape(B, T, -1)              # [B, T, d_model]

        # --- Add positional encoding ---
        tokens = tokens + self.pos_emb[:, :T, :]

        # --- Normalize before attention ---
        tokens = self.input_norm(tokens)

        # --- Transformer ---
        struct = self.encoder(tokens)             # [B, T, d_model]

        return struct