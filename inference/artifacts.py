import os

import torch
import matplotlib.pyplot as plt

from SingLS.config.config import EXP_PATH, hidden_size, DEVICE, AttentionType, lr
from SingLS.model.model import MusicGenerator
from SingLS.trainer.train import ModelTrainer
from inference.compare_models import piano_roll_to_midi, compute_ssm


def generate_artifacts(model_path, attention_type, data_path, length=95, save_dir="generated"):
    os.makedirs(save_dir, exist_ok=True)

    # === Загрузка данных ===
    full_data_path = os.path.join(EXP_PATH, data_path)
    data = torch.load(full_data_path, weights_only=False)

    # === Модель ===
    model = MusicGenerator(128, 128, attention_type=attention_type).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ModelTrainer(model, optimizer, data, hidden_size)

    # === Генерация ===
    snap = trainer.generate_n_examples(n=1, length=length, starter_notes=10)
    piano_roll = snap.squeeze(1).detach().cpu().numpy().round()[:length]  # [T, 128]

    # === MIDI ===
    midi = piano_roll_to_midi(piano_roll)
    midi.write(os.path.join(save_dir, "music_sample.mid"))

    # === Piano Roll ===
    plt.figure(figsize=(12, 4))
    plt.imshow(piano_roll.T, aspect='auto', origin='lower', cmap='gray_r')
    plt.title("Piano Roll (Generated)")
    plt.savefig(os.path.join(save_dir, "pianoroll.png"))
    plt.close()

    # === Reference ===
    reference_roll = torch.stack([x for x in data[0][:length] if isinstance(x, torch.Tensor)])
    reference_roll = reference_roll.to(torch.float32).cpu().numpy()
    if reference_roll.ndim == 3 and reference_roll.shape[0] == 1:
        reference_roll = reference_roll.squeeze(0)
    reference_roll = reference_roll[:length]

    # === Self-Similarity Matrices ===
    ssm_gen = compute_ssm(piano_roll)
    ssm_ref = compute_ssm(reference_roll)

    plt.figure(figsize=(6, 6))
    plt.imshow(ssm_gen, origin='lower', cmap='hot')
    plt.title("SSM (Generated)")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, "ssm_generated.png"))
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(ssm_ref, origin='lower', cmap='hot')
    plt.title("SSM (Reference)")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, "ssm_reference.png"))
    plt.close()

    print(f"Saved all artifacts to: {save_dir}")


if __name__ == "__main__":
    # data_ = "mar-1-variable_bin_bounds_test.csv"
    data_ = "mar-1-variable_bin_bounds_val.csv"
    len_ = 105
    # ORIGINAL
    generate_artifacts(
        model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/SingLS/models/checkpoints_original_15e/model-epoch-15-loss-16.65498.pt",
        attention_type=AttentionType.ORIGINAL,
        data_path=data_,
        length=len_,
        save_dir="generated")
    # LSA
    generate_artifacts(
        model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/SingLS/models/checkpoints_lsa_15e/model-epoch-15-loss-13.02090.pt",
        attention_type=AttentionType.LSA,
        data_path=data_,
        length=len_,
        save_dir="generated_lsa")
    # ORIGINAL alpha
    generate_artifacts(
        model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/SingLS/models/checkpoints_original_15e_alpha/model-epoch-15-loss-13.71772.pt",
        attention_type=AttentionType.ORIGINAL,
        data_path=data_,
        length=len_,
        save_dir="generated_alpha")
    # LSA alpha
    generate_artifacts(
        model_path="/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/SingLS/models/checkpoints_lsa_alpha_15e/model-epoch-15-loss-6.11968.pt",
        attention_type=AttentionType.LSA,
        data_path=data_,
        length=len_,
        save_dir="generated_lsa_alpha")
