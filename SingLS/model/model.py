import torch
import torch.nn as nn
import sparsemax
from SingLS.config.config import DEVICE, AttentionType
from SingLS.model.lsa import LearnedStructuredAttention
from SingLS.model.lsaSB import LearnedStructuredAttentionSB
from SingLS.model.original_attention import original_attention
from SingLS.model.utils import build_ssm_batch


class MusicGenerator(nn.Module):
    def __init__(self, hidden_size, output_size, attention_type='original'):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_type = attention_type

        self.lstm = nn.LSTM(output_size, hidden_size, num_layers=1)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None

        self.attention = nn.Linear(2, 1)
        self.softmax = sparsemax.Sparsemax(dim=1)

        self.lsa = LearnedStructuredAttention(hidden_size)
        self.lsa_sb = LearnedStructuredAttentionSB(hidden_size)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_size).float().to(DEVICE)
        self.hidden = (hidden, hidden)

    def set_random_hidden(self, batch_size):
        hidden = torch.randn(1, batch_size, self.hidden_size).float().to(DEVICE)
        self.hidden = (hidden, hidden)

    def _apply_attention(self, output, prev_sequence, batched_ssm):
        # output: [T,B,H]
        avg_output = output[-1, :, :].unsqueeze(1)  # [B,1,H]

        if self.attention_type == AttentionType.NONE:
            return avg_output.transpose(0, 1)  # [1,B,H]

        if self.attention_type == AttentionType.ORIGINAL:
            return original_attention(
                output, prev_sequence, batched_ssm,
                attention_layer=self.attention,
                sparsemax=self.softmax,
                device=output.device
            )

        if self.attention_type == AttentionType.LSA:
            T, B = output.shape[0], output.shape[1]
            ssm_batch = build_ssm_batch(T, B, batched_ssm, output.device)
            return self.lsa(output, ssm_batch)

        if self.attention_type == AttentionType.LSA_SB:
            T, B = output.shape[0], output.shape[1]
            ssm_batch = build_ssm_batch(T, B, batched_ssm, output.device)
            return self.lsa_sb(output, ssm_batch)

        raise ValueError(f"Unknown attention type: {self.attention_type}")

    def forward(self, in_put, batch_size, prev_sequence, batched_ssm, struct=None, return_lstm_output=False):
        output, self.hidden = self.lstm(in_put.float().to(DEVICE), self.hidden)  # [T,B,H]

        if struct is not None:
            output = output + struct

        attn_output = self._apply_attention(output, prev_sequence, batched_ssm)

        if return_lstm_output:
            return attn_output, self.hidden, output
        return attn_output, self.hidden
