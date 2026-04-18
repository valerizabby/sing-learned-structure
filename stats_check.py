import torch, sys
sys.path.insert(0, '.')
from SingLS.model.model import MusicGenerator
from SingLS.model.hierarchical_generator import HierarchicalGenerator
from SingLS.model.structure_transformer import StructureTransformer, StructureModel
from SingLS.config.config import hidden_size, output_size, AttentionType

# LSTM-only model
lstm = MusicGenerator(hidden_size, output_size, AttentionType.ORIGINAL)
lstm_params = sum(p.numel() for p in lstm.parameters())
print(f'LSTM-only (MusicGenerator): {lstm_params:,} params')

# HierarchicalGenerator
gen = MusicGenerator(hidden_size, output_size, AttentionType.ORIGINAL)
st = StructureTransformer(d_model=hidden_size, nhead=4, num_layers=2)
sm = StructureModel(st, torch.nn.Linear(hidden_size, hidden_size))
hier = HierarchicalGenerator(gen, sm)
hier_params = sum(p.numel() for p in hier.parameters())
print(f'HierarchicalGenerator (LSTM+Transformer): {hier_params:,} params')

# Best model
best = torch.load('data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt', weights_only=False, map_location='cpu')
best_params = sum(p.numel() for p in best.parameters())
print(f'Best model (loaded): {best_params:,} params, type={type(best).__name__}')
print(f'Best model generator type: {type(best.generator).__name__}')
print(f'Best model attention: {best.generator.attention_type}')
print(f'Best model structure_model: {best.structure_model}')

# Hypothetical pure Transformer replacing LSTM
# d_model=128, nhead=4, num_layers=2, ff=512, seq_len=700
d_model = 128
nhead = 4
num_layers = 2
ff = 512

import torch.nn as nn
enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff, batch_first=True)
encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
proj = nn.Linear(d_model, 128)
transformer_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in proj.parameters())
print(f'\nHypothetical pure Transformer (2L, d=128, ff=512): {transformer_params:,} params')

# Data stats
data = torch.load('data/combined/combined_train.pt', weights_only=False)
print(f'\nTrain samples: {len(data)}')
lengths = [d[2] for d in data]
import statistics
print(f'Sequence lengths (beats): min={min(lengths)}, max={max(lengths)}, median={statistics.median(lengths)}, mean={statistics.mean(lengths):.1f}')
# Total tokens for quadratic attention
import math
total_tokens = sum(lengths)
print(f'Total beats: {total_tokens:,}')
print(f'Avg quadratic attention cost per sample (T^2): {statistics.mean([l**2 for l in lengths]):,.0f}')
print(f'Max quadratic attention cost (T^2 at T=700): {700**2:,}')

