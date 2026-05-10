import torch
import torch.nn as nn
from SingLS.model.utils import build_ssm_batch

class HierarchicalGenerator(nn.Module):
    def __init__(self, generator, structure_model=None):
        super().__init__()
        self.generator = generator
        self.structure_model = structure_model
        self.register_buffer("alpha", torch.tensor(0.05))

    def forward(self, in_put, batch_size, prev_sequence, batched_ssm):
        attn_out, hidden, lstm_out = self.generator.forward(
            in_put, batch_size, prev_sequence, batched_ssm,
            struct=None,
            return_lstm_output=True
        )

        if self.structure_model is None:
            return attn_out, hidden

        # Используем длину контекста (prev_sequence), а не длину шага LSTM (T=1
        # при авторегрессии). Иначе build_ssm_batch всегда берёт top-left 1×1
        # угол SSM, что бессмысленно.
        T = max(lstm_out.shape[0], prev_sequence.shape[0])
        ssm_batch = build_ssm_batch(
            output_len=T,
            batch_size=batch_size,
            batched_ssm=batched_ssm,
            device=lstm_out.device
        )

        struct = self.structure_model(ssm_batch)   # [T, B, H]
        # Align struct to lstm_out: T = max(lstm_T, prev_T) >= lstm_T,
        # so we take the last lstm_T rows (positionally correct for warmup).
        struct = struct[-lstm_out.shape[0]:]       # [lstm_T, B, H]
        mixed = lstm_out + self.alpha * struct

        attn_out2 = self.generator._apply_attention(mixed, prev_sequence, batched_ssm)
        return attn_out2, hidden

    @property
    def attention_type(self):
        return self.generator.attention_type

    def init_hidden(self, batch_size):
        return self.generator.init_hidden(batch_size)

    def set_random_hidden(self, batch_size):
        return self.generator.set_random_hidden(batch_size)