import os
from enum import Enum

import torch

DEVICE = torch.device("cpu")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_PATH = os.path.join(_ROOT, "data")
EXP_PATH_LMD = os.path.join(_ROOT, "data", "lmd_processed")
EXP_PATH_COMBINED = os.path.join(_ROOT, "data", "combined")


hidden_size = 128
num_epochs = 30
output_size = 128
lr = 0.001
struct_lr = 0.0001

class AttentionType(Enum):
    NONE = "none"
    ORIGINAL = "original"
    LSA = "lsa"
    LSA_SB = "lsa_sb"