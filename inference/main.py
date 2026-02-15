import os

from SingLS.config.config import AttentionType, EXP_PATH_LMD, EXP_PATH, EXP_PATH_COMBINED
from inference.compare_models import compare_models_avg

if __name__ == "__main__":
    length = 95
    n = 10

    ious1 = []
    ious2 = []

    mses1 = []
    mses2 = []

    data_path_ = os.path.join(EXP_PATH_COMBINED, "combined_test.pt")
    for i in range(20):
        metrics_original = compare_models_avg(
            model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data/meta_info/trained_original_combined/model_30_epochs.txt",
            data_path=data_path_,
            _len_=length,
            n = 30
        )
        metrics_transformer = compare_models_avg(
            model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data/meta_info/trained_lsa_combined/model_30_epochs.txt",
            data_path=data_path_,
            _len_=length,
            n = 30
        )
        ious1.append(metrics_original['iou'])
        ious2.append(metrics_transformer["iou"])

        mses1.append(metrics_original["mse"])
        mses2.append(metrics_transformer["mse"])

    print("\n=== AVERAGED METRICS ===")

    print(f"IOU: ORIGINAL={sum(ious1) / len(ious1):.4f}, TRANSFORMER={sum(ious2) / len(ious2):.4f}")
    print(f"MSE: ORIGINAL={sum(mses1) / len(mses1):.4f}, TRANSFORMER={sum(mses2) / len(mses2):.4f}")
