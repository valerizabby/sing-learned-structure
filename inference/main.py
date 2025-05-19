from SingLS.config.config import AttentionType
from inference.compare_models import compare_models_avg

if __name__ == "__main__":
    length = 95
    n = 10

    # Загрузка данных
    data_path_ = "mar-1-variable_bin_bounds_val.csv"

    # Модель 1: ORIGINAL
    metrics_original = compare_models_avg(
        model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/SingLS/models/checkpoints_original_15e/model-epoch-15-loss-16.65498.pt",
        attention_type_=AttentionType.ORIGINAL,
        data_path_=data_path_,
        _len_=length,
        n = 20
    )

    # Модель 2: LSA
    metrics_lsa = compare_models_avg(
        model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/SingLS/models/checkpoints/model-epoch-15-loss-13.02090.pt",
        attention_type_=AttentionType.ORIGINAL,
        data_path_=data_path_,
        _len_=length,
        n = 20
    )

    print("\n=== AVERAGED METRICS ===")
    for key in metrics_original.keys():
        print(f"{key}: ORIGINAL={metrics_original[key]:.4f}, LSA={metrics_lsa[key]:.4f}")
