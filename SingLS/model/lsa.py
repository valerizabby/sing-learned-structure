import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedStructuredAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x, ssm):
        # x: [seq_len, batch, hidden]
        # ssm: [batch, seq_len, seq_len]

        Q = self.query_proj(x)       # [seq_len, batch, hidden]
        K = self.key_proj(x)
        V = self.value_proj(x)

        Q = Q.transpose(0, 1)  # [batch, seq_len, hidden]
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [batch, seq_len, seq_len]
        masked_scores = attn_scores * ssm  # Element-wise modulation by SSM
        attn_weights = F.softmax(masked_scores, dim=-1)  # [batch, seq_len, seq_len]
        output = torch.bmm(attn_weights, V)  # [batch, seq_len, hidden]

        return output.transpose(0, 1)  # [seq_len, batch, hidden]