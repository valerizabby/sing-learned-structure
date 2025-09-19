
import os
from typing import Optional

import numpy as np
import pandas as pd
import pretty_midi
import torch
from tqdm import tqdm
from bisect import bisect
import math

# === Настройки ===
CSV_PATH = "tracks_with_aligned_midi.csv"  # путь к CSV с метаданными
MIDI_ROOT = "."  # путь относительно которого указаны midi_filename
OUTPUT_DIR = "usable_data"
NUM_BINS = 16
MAX_LENGTH = 700


# === Класс бин ===
class Bin:
    def __init__(self, length, lower_bound, upper_bound):
        self.bin_length = length
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.pieces = []
        self.bin_count = 0

    def add_piece(self, roll, length):
        self.bin_count += 1
        if length >= self.bin_length:
            roll = roll[:self.bin_length]
        else:
            roll = np.vstack((roll, roll))[:self.bin_length]
        self.pieces.append(roll)
        return roll

    def add_sub_piece(self, roll, length, roll_before, roll_after):
        self.bin_count += 1
        beats_to_add = self.bin_length - length
        if roll_before is None and roll_after is not None:
            beats_after = roll_after[:beats_to_add]
            roll = np.vstack((roll, beats_after))
        elif roll_after is None and roll_before is not None:
            beats_before = roll_before[-beats_to_add:]
            roll = np.vstack((beats_before, roll))
        else:
            half = beats_to_add // 2
            beats_before = roll_before[-half:]
            beats_after = roll_after[:beats_to_add - half]
            roll = np.vstack((beats_before, roll, beats_after))
        self.pieces.append(roll)
        return roll


# === Хранение всех бин ===
class BinHolder:
    def __init__(self, num_bins, min_size, max_size):
        self.bins = []
        self.num_bins = num_bins
        self.min_size = min_size
        self.max_size = max_size
        self.a = min_size * (min_size / max_size)**(1/(num_bins-1))
        self.b = (1/(num_bins-1)) * math.log(max_size/min_size)

        for i in range(1, num_bins+1):
            lower = self._exp(i - 0.5)
            length = self._exp(i)
            upper = self._exp(i + 0.5)
            self.bins.append(Bin(length, lower, upper))

        self.bin_bounds = [self._exp(i + 0.5) for i in range(num_bins+1)]
        self.max_bin_bound = self.bin_bounds[-1]

    def _exp(self, x):
        return int(self.a * math.exp(self.b * x))

    def get_bin_index(self, length):
        return bisect(self.bin_bounds, length) - 1

    def add_piece(self, roll, length):
        fit_rolls = []
        if length >= self.max_bin_bound:
            n_splits = math.ceil(length / self.max_size)
            pieces = np.array_split(roll, n_splits)
            for i in range(n_splits):
                before = pieces[i-1] if i > 0 else None
                after = pieces[i+1] if i < len(pieces)-1 else None
                bin_index = self.get_bin_index(pieces[i].shape[0])
                fit_roll = self.bins[bin_index].add_sub_piece(pieces[i], pieces[i].shape[0], before, after)
                fit_rolls.append(fit_roll)
        else:
            bin_index = self.get_bin_index(length)
            fit_roll = self.bins[bin_index].add_piece(roll, length)
            fit_rolls.append(fit_roll)
        return fit_rolls


# === Загрузка и обработка ===
def piano_roll_no_silence(midi_path: str) -> Optional[tuple[np.ndarray, float, int]]:
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        tempo = midi.estimate_tempo()
        roll = midi.get_piano_roll(tempo / 60).T

        # Remove silence at beginning
        while len(roll) > 0 and np.array_equal(roll[0], np.zeros(128)):
            roll = roll[1:]

        # Remove silence at end
        while len(roll) > 0 and np.array_equal(roll[-1], np.zeros(128)):
            roll = roll[:-1]

        if len(roll) == 0:
            return None  # трек пустой после удаления тишины

        roll = np.where(roll > 0, 1, 0)
        beats = roll.shape[0]
        return roll, tempo, beats
    except Exception as e:
        print(f"[!] Skipping {midi_path} due to error: {e}")
        return None


def add_roll_to_split(container, roll, tempo, beats):
    container.append(np.array([torch.tensor(roll), tempo, beats], dtype="object"))


def main():
    df = pd.read_parquet("tracks_with_aligned_midi.parquet")
    all_beats = []

    pieces = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        midi_path = os.path.join(MIDI_ROOT, row["midi_filename"])
        split = row["split"]
        result = piano_roll_no_silence(midi_path)
        if result is None:
            continue  # skip this piecef
        roll, tempo, beats = result
        pieces.append({
            "roll": roll,
            "tempo": tempo,
            "beats": beats,
            "split": split
        })
        all_beats.append(beats)

    all_beats.sort()
    min_len = all_beats[9]  # 10-й по длине

    bin_holder = BinHolder(NUM_BINS, min_len, MAX_LENGTH)

    train, test, validation = [], [], []

    for piece in pieces:
        roll = piece["roll"]
        tempo = piece["tempo"]
        beats = piece["beats"]
        split = piece["split"]
        fit_rolls = bin_holder.add_piece(roll, beats)
        for fr in fit_rolls:
            if split == "train":
                add_roll_to_split(train, fr, tempo, fr.shape[0])
            elif split == "test":
                add_roll_to_split(test, fr, tempo, fr.shape[0])
            else:
                add_roll_to_split(validation, fr, tempo, fr.shape[0])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(train, os.path.join(OUTPUT_DIR, "lmd_train.pt"))
    torch.save(test, os.path.join(OUTPUT_DIR, "lmd_test.pt"))
    torch.save(validation, os.path.join(OUTPUT_DIR, "lmd_val.pt"))


if __name__ == "__main__":
    main()
