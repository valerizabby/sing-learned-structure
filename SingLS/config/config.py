from enum import Enum

import torch

DEVICE = torch.device("cpu")
EXP_PATH = "/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data"
EXP_PATH_LMD = "/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data/lmd_processed"
EXP_PATH_COMBINED = "/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data/combined"


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