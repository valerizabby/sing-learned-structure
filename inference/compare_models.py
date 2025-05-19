import os
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi

from SingLS.config.config import AttentionType, DEVICE, EXP_PATH, lr, hidden_size
from SingLS.model.model import MusicGenerator
from SingLS.trainer.train import ModelTrainer
from SingLS.trainer.data_utils import get_chroma


def piano_roll_to_midi(piano_roll, fs=100, program=0):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    piano_roll = piano_roll.T  # [128, T]
    notes, frames = piano_roll.shape
    velocity = 100

    for note in range(notes):
        on = None
        for t in range(frames):
            if piano_roll[note, t] > 0 and on is None:
                on = t
            elif piano_roll[note, t] == 0 and on is not None:
                start = on / fs
                end = t / fs
                instrument.notes.append(pretty_midi.Note(velocity, note, start, end))
                on = None
        if on is not None:
            instrument.notes.append(pretty_midi.Note(velocity, note, on / fs, frames / fs))

    midi.instruments.append(instrument)
    return midi


def compute_ssm(sequence, block_size=4):
    chroma = get_chroma(torch.tensor(sequence, dtype=torch.float32), sequence.shape[0])  # [T, 12]
    chroma = chroma / (chroma.norm(dim=1, keepdim=True) + 1e-8)
    ssm = torch.matmul(chroma, chroma.T).cpu().numpy()

    np.fill_diagonal(ssm, 0)

    T = ssm.shape[0]
    pad = (block_size - T % block_size) % block_size
    if pad > 0:
        ssm = np.pad(ssm, ((0, pad), (0, pad)), mode='constant')

    new_size = ssm.shape[0] // block_size
    ssm_block = ssm.reshape(new_size, block_size, new_size, block_size).mean(axis=(1, 3))
    return ssm_block


def iou_ssm(ssm1, ssm2, threshold=0.5):
    bin1 = ssm1 > threshold
    bin2 = ssm2 > threshold
    intersection = np.logical_and(bin1, bin2).sum()
    union = np.logical_or(bin1, bin2).sum()
    return intersection / (union + 1e-8)


def mse_ssm(ssm1, ssm2):
    if ssm1.shape != ssm2.shape:
        min_size = min(ssm1.shape[0], ssm2.shape[0])
        ssm1 = ssm1[:min_size, :min_size]
        ssm2 = ssm2[:min_size, :min_size]
    return np.mean((ssm1 - ssm2) ** 2)


def evaluate_piano_roll(piano_roll, reference_ssm=None, block_size=4):
    ssm = compute_ssm(piano_roll, block_size=block_size)

    diversity_score = 1.0 - ssm.mean()
    unique_steps = np.unique(piano_roll, axis=0).shape[0]
    mean_ssm = ssm.mean()
    std_ssm = ssm.std()
    diag_vals = np.diag(ssm)
    off_diag_vals = ssm[~np.eye(ssm.shape[0], dtype=bool)]
    diag_drop = 1.0 - diag_vals.mean() / (off_diag_vals.mean() + 1e-8)

    if reference_ssm is not None:
        iou = iou_ssm(ssm, reference_ssm)
        mse = mse_ssm(ssm, reference_ssm)
    else:
        iou, mse = None, None

    return {
        "diversity": diversity_score,
        "unique_steps": unique_steps,
        "mean_ssm": mean_ssm,
        "std_ssm": std_ssm,
        "diag_drop": diag_drop,
        "iou": iou,
        "ssm": ssm,
        "mse": mse,
    }


def compare_models(model_path,
                   attention_type_=AttentionType.ORIGINAL,
                   data_path_="mar-1-variable_bin_bounds_val.csv",
                   _len_=95
                   ):
    data_path = os.path.join(EXP_PATH, data_path_)
    data = torch.load(data_path, weights_only=False)

    model = MusicGenerator(128, 128, attention_type=attention_type_).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ModelTrainer(model, optimizer, data, hidden_size)
    snap = trainer.generate_n_examples(n=1, length=_len_, starter_notes=10)

    piano_roll = snap.squeeze(1).detach().cpu().numpy().round()
    # обрежь до нужной длины
    piano_roll = piano_roll[:_len_]

    os.makedirs("generated", exist_ok=True)
    midi = piano_roll_to_midi(piano_roll)
    midi.write("generated/music_sample.mid")

    plt.figure(figsize=(12, 4))
    plt.imshow(piano_roll.T, aspect='auto', origin='lower', cmap='gray_r')
    plt.title("Piano Roll (Generated)")
    plt.savefig("generated/pianoroll.png")
    plt.close()

    # reference (например, первый элемент из train)
    reference_roll = torch.stack([x for x in data[0][:_len_] if isinstance(x, torch.Tensor)])
    reference_roll = reference_roll.to(torch.float32).cpu().numpy()  # [95, 128]

    print("reference_roll.shape before slicing:", reference_roll.shape)
    print("piano_roll.shape:", piano_roll.shape)

    if reference_roll.ndim == 3 and reference_roll.shape[0] == 1:
        reference_roll = reference_roll.squeeze(0)

    # обрезка до длины тарегта
    target_length = piano_roll.shape[0]  # T сгенерированной последовательности
    reference_roll = reference_roll[:target_length]  # [T, 128]
    reference_ssm = compute_ssm(reference_roll)

    # Метрики
    metrics = evaluate_piano_roll(piano_roll, reference_ssm=reference_ssm)
    ssm = metrics["ssm"]

    plt.figure(figsize=(6, 6))
    plt.imshow(ssm, origin='lower', cmap='hot')
    plt.title("SSM (block-aggregated)")
    plt.colorbar()
    plt.savefig("generated/ssm_plot.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(reference_ssm, origin='lower', cmap='hot')
    plt.title("SSM (Reference)")
    plt.colorbar()
    plt.savefig("generated/ssm_reference.png")
    plt.close()

    # Печать метрик
    print("\n===== STRUCTURE METRICS =====")
    for k, v in metrics.items():
        if k == "ssm":
            continue
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


def compare_models_avg(model_path, attention_type_, data_path_, _len_=95, n=10):
    data_path = os.path.join(EXP_PATH, data_path_)
    data = torch.load(data_path, weights_only=False)

    reference_roll = torch.stack([x for x in data[0][:_len_] if isinstance(x, torch.Tensor)])
    reference_roll = reference_roll.to(torch.float32).cpu().numpy()
    if reference_roll.ndim == 3 and reference_roll.shape[0] == 1:
        reference_roll = reference_roll.squeeze(0)
    reference_roll = reference_roll[:_len_]
    reference_ssm = compute_ssm(reference_roll)

    model = MusicGenerator(128, 128, attention_type=attention_type_).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ModelTrainer(model, optimizer, data, hidden_size)

    all_metrics = defaultdict(list)
    for _ in range(n):
        snap = trainer.generate_n_examples(n=1, length=_len_, starter_notes=10)
        piano_roll = snap.squeeze(1).detach().cpu().numpy().round()
        piano_roll = piano_roll[:_len_]
        metrics = evaluate_piano_roll(piano_roll, reference_ssm=reference_ssm)
        for k, v in metrics.items():
            if k != "ssm":
                all_metrics[k].append(v)

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}
