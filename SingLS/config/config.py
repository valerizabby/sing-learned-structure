from enum import Enum

import torch

DEVICE = torch.device("cpu")
EXP_PATH = "/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data"


hidden_size = 128
num_epochs = 15
output_size = 128
lr = 0.001

class AttentionType(Enum):
    NONE = "none"
    ORIGINAL = "original"
    LSA = "lsa"