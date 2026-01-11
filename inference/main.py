import os

from SingLS.config.config import AttentionType, EXP_PATH_LMD, EXP_PATH, EXP_PATH_COMBINED
from inference.compare_models import compare_models_avg

if __name__ == "__main__":
    length = 95
    n = 10

    # Загрузка данных
    # COMBINED data
    data_path_ = os.path.join(EXP_PATH_COMBINED, "combined_test.pt")

    # Модель 1: LSA combined data + alpha
    metrics_original = compare_models_avg(
        # model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data/meta_info/trained_lsa_combined/model_30_epochs.txt",
        # model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data/meta_info/trained_none_combined/model_30_epochs.txt",
        # model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data/meta_info/trained_original_combined/model_30_epochs.txt",
        model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data/meta_info/trained_original_combined/model_30_epochs.txt",
        data_path=data_path_,
        _len_=length,
        n = 30
    )
    # Модель 2: LSA combined data
    metrics_lsa = compare_models_avg(
        # model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data/meta_info/trained_lsa_combined_NONalpha/model_30_epochs.txt",
        # model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data/meta_info/trained_none_combined_NONalpha/model_30_epochs.txt",
        model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data/meta_info/trained_transformer_original_combined/model_30_epochs.txt",
        data_path=data_path_,
        _len_=length,
        n = 30
    )

    print("\n=== AVERAGED METRICS ===")
    for key in metrics_original.keys():
        print(f"{key}: ORIGINAL={metrics_original[key]:.4f}, TRANSFORMER={metrics_lsa[key]:.4f}")
