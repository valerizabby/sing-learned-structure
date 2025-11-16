import torch
import random
import os

from SingLS.config.config import EXP_PATH_LMD, EXP_PATH, EXP_PATH_COMBINED

LMD_PATH_TRAIN = os.path.join(EXP_PATH_LMD, "lmd_train.pt")
LMD_PATH_VAL = os.path.join(EXP_PATH_LMD, "lmd_val.pt")
LMD_PATH_TEST = os.path.join(EXP_PATH_LMD, "lmd_test.pt")

MAESTRO_PATH_TRAIN = os.path.join(EXP_PATH, "mar-1-variable_bin_bounds_train.csv")
MAESTRO_PATH_VAL = os.path.join(EXP_PATH, "mar-1-variable_bin_bounds_val.csv")
MAESTRO_PATH_TEST = os.path.join(EXP_PATH, "mar-1-variable_bin_bounds_test.csv")

OUTPUT_PATH = lambda val: os.path.join(EXP_PATH_COMBINED, val)

def combine(path1, path2, out):
    # Загрузка данных
    data_lmd = torch.load(path1, weights_only=False)
    data_maestro = torch.load(path2, weights_only=False)

    print(f"LMD samples: {len(data_lmd)}, Maestro samples: {len(data_maestro)}")

    # Объединение
    combined = list(data_lmd) + list(data_maestro)

    # Перемешивание
    random.shuffle(combined)

    # Сохранение
    torch.save(combined, out)
    print(f"Saved combined dataset: {OUTPUT_PATH} ({len(combined)} samples)")

if __name__ == "__main__":
    combine(LMD_PATH_TRAIN, MAESTRO_PATH_TRAIN, OUTPUT_PATH("combined_train.pt"))
    combine(LMD_PATH_TEST, MAESTRO_PATH_TEST, OUTPUT_PATH("combined_test.pt"))
    combine(LMD_PATH_VAL, MAESTRO_PATH_VAL, OUTPUT_PATH("combined_val.pt"))