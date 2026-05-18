import argparse
import os

from SingLS.config.config import EXP_PATH_COMBINED
from inference.compare_models import compare_models_avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two model checkpoints on combined_test")
    parser.add_argument("--model1", required=True, help="Path to first model checkpoint")
    parser.add_argument("--model2", required=True, help="Path to second model checkpoint")
    parser.add_argument("--data", default=os.path.join(EXP_PATH_COMBINED, "combined_test.pt"))
    parser.add_argument("--length", type=int, default=95)
    parser.add_argument("--n", type=int, default=30, help="Runs per model for averaging")
    parser.add_argument("--repeats", type=int, default=20, help="Outer repetitions")
    args = parser.parse_args()

    ious1, ious2 = [], []
    mses1, mses2 = [], []

    for _ in range(args.repeats):
        m1 = compare_models_avg(args.model1, args.data, _len_=args.length, n=args.n)
        m2 = compare_models_avg(args.model2, args.data, _len_=args.length, n=args.n)
        ious1.append(m1["iou"])
        ious2.append(m2["iou"])
        mses1.append(m1["mse"])
        mses2.append(m2["mse"])

    print("\n=== AVERAGED METRICS ===")
    print(f"IOU: model1={sum(ious1)/len(ious1):.4f}, model2={sum(ious2)/len(ious2):.4f}")
    print(f"MSE: model1={sum(mses1)/len(mses1):.4f}, model2={sum(mses2)/len(mses2):.4f}")
