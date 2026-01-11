import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedStructuredAttentionSB(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # learnable β (initialized to 1.0)
        self.beta = nn.Parameter(torch.tensor(1.0))

        self.scale = hidden_dim ** 0.5

    def forward(self, x, ssm):
        """
        x:   [seq_len, batch, hidden]
        ssm: [batch, seq_len, seq_len]
        """

        # Project QKV
        Q = self.query_proj(x)  # [seq, B, H]
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Move seq dimension to the end
        Q = Q.transpose(0, 1)  # [B, seq, H]
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)

        # Base attention logits
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, seq, seq]

        # ADD structural bias instead of masking
        attn_scores = attn_scores + self.beta * ssm

        # Softmax over last dimension
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted values
        output = torch.bmm(attn_weights, V)  # [B, seq, H]

        return output.transpose(0, 1)  # [seq, B, H]