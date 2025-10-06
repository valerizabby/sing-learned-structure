import torch

if __name__ == "__main__":
    data = torch.load("lmd_test.pt", weights_only=False)

    for i, (roll, tempo, beats) in enumerate(data):
        if not torch.isfinite(roll).all():
            print(f"Sample {i} contains NaN or Inf")
        if roll.shape[0] < 2:
            print(f"Sample {i} is too short: {roll.shape[0]} beats")
        if roll.sum() == 0:
            print(f"Sample {i} is all zeros")
        print(f"{roll.shape[0]} x {roll.shape[1]}")