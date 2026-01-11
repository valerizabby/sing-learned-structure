from enum import Enum

from SingLS.config.config import AttentionType


class Model(Enum):
    LSA_SB = (AttentionType.LSA_SB, "lsa_sb") # +
    LSA = (AttentionType.LSA, "lsa")
    ORIGINAL = (AttentionType.ORIGINAL, "original")
    NONE = (AttentionType.NONE, "none")

    TRANSFORMER_LSA_SB = (AttentionType.LSA_SB, "transformer_lsa_sb") # +
    TRANSFORMER_LSA = (AttentionType.LSA, "transformer_lsa")
    TRANSFORMER_ORIGINAL = (AttentionType.ORIGINAL, "transformer_original")
    TRANSFORMER = (AttentionType.NONE, "transformer")

if __name__ == '__main__':
    # print("Model.ORIGINAL" + Model.ORIGINAL.)
    print(Model.ORIGINAL.value)
    print("Model.ORIGINAL.value[0]" + Model.TRANSFORMER_ORIGINAL.value[0].value)
    print("Model.ORIGINAL.value[1]" + Model.TRANSFORMER_ORIGINAL.value[1])
