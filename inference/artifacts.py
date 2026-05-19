import argparse
import os

import torch
import matplotlib.pyplot as plt

from SingLS.config.config import EXP_PATH, hidden_size, DEVICE, AttentionType, lr
from SingLS.model.model import MusicGenerator
from SingLS.trainer.train import ModelTrainer
from inference.compare_models import compute_ssm
from pipeline.generate import piano_roll_to_midi


def generate_artifacts(model_path, attention_type, data_path, length=95, save_dir="generated"):
    os.makedirs(save_dir, exist_ok=True)

    data = torch.load(os.path.join(EXP_PATH, data_path), weights_only=False)

    model = MusicGenerator(128, 128, attention_type=attention_type).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ModelTrainer(model, optimizer, data, hidden_size)

    snap = trainer.generate_n_examples(n=1, length=length, starter_notes=10)
    piano_roll = snap.squeeze(1).detach().cpu().numpy().round()[:length]

    midi = piano_roll_to_midi(piano_roll, tempo=120)
    midi.write(os.path.join(save_dir, "music_sample.mid"))

    plt.figure(figsize=(12, 4))
    plt.imshow(piano_roll.T, aspect="auto", origin="lower", cmap="gray_r")
    plt.title("Piano Roll (Generated)")
    plt.savefig(os.path.join(save_dir, "pianoroll.png"))
    plt.close()

    reference_roll = torch.stack([x for x in data[0][:length] if isinstance(x, torch.Tensor)])
    reference_roll = reference_roll.to(torch.float32).cpu().numpy()
    if reference_roll.ndim == 3 and reference_roll.shape[0] == 1:
        reference_roll = reference_roll.squeeze(0)
    reference_roll = reference_roll[:length]

    ssm_gen = compute_ssm(piano_roll)
    ssm_ref = compute_ssm(reference_roll)

    for ssm, title, fname in [
        (ssm_gen, "SSM (Generated)", "ssm_generated.png"),
        (ssm_ref, "SSM (Reference)", "ssm_reference.png"),
    ]:
        plt.figure(figsize=(6, 6))
        plt.imshow(ssm, origin="lower", cmap="hot")
        plt.title(title)
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, fname))
        plt.close()

    print(f"Saved all artifacts to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualisation artifacts for a model checkpoint")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--attention", default="original",
                        choices=[t.value for t in AttentionType], help="Attention type")
    parser.add_argument("--data", required=True, help="Relative path to data file under EXP_PATH")
    parser.add_argument("--length", type=int, default=95)
    parser.add_argument("--out_dir", default="generated")
    args = parser.parse_args()

    attention_type = AttentionType(args.attention)
    generate_artifacts(args.model, attention_type, args.data, args.length, args.out_dir)
