import torch
import torch.nn as nn
import sparsemax
from SingLS.config.config import DEVICE, AttentionType
from SingLS.model.lsa import LearnedStructuredAttention
from SingLS.model.original_attention import original_attention


class MusicGenerator(nn.Module):
    def __init__(self, hidden_size, output_size, attention_type='original'):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_type = attention_type  # 'none', 'original', 'lsa'

        self.lstm = nn.LSTM(output_size, hidden_size, num_layers=1)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None

        # only if original attention is used
        self.attention = nn.Linear(2, 1)
        self.softmax = sparsemax.Sparsemax(dim=1)

        # if LSA
        self.lsa = LearnedStructuredAttention(hidden_size)


    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_size).float().to(DEVICE)
        self.hidden = (hidden, hidden)

    def set_random_hidden(self, batch_size):
        hidden = torch.randn(1, batch_size, self.hidden_size).float().to(DEVICE)
        self.hidden = (hidden, hidden)

    def forward(self, in_put, batch_size, prev_sequence, batched_ssm):
        output, self.hidden = self.lstm(in_put.float().to(DEVICE), self.hidden)

        avg_output = output[-1, :, :].unsqueeze(1)

        if self.attention_type == AttentionType.NONE:
            return avg_output.transpose(0, 1), self.hidden

        if self.attention_type == AttentionType.ORIGINAL:
            attn_output = original_attention(
                output, prev_sequence, batched_ssm,
                attention_layer=self.attention,
                sparsemax=self.softmax,
                device=DEVICE
            )
            return attn_output, self.hidden

        if self.attention_type == AttentionType.LSA:
            if self.attention_type == AttentionType.LSA:
                T = output.shape[0]
                B = output.shape[1]
                SSM = []
                for i in range(B):
                    offset = i * batched_ssm.shape[1]
                    ssm_slice = batched_ssm[offset:offset + T, :T]  # [T, T]
                    SSM.append(ssm_slice.unsqueeze(0))  # [1, T, T]
                ssm_batch = torch.cat(SSM, dim=0).to(output.device)  # [B, T, T]

                attn_output = self.lsa(output, ssm_batch)
                return attn_output, self.hidden

        raise ValueError(f"Unknown attention type: {self.attention_type}")