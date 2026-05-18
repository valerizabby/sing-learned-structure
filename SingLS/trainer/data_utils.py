import numpy as np
import torch
import torch.nn.functional as F
import random
import torch.distributions
import sparsemax
import math

MIDI_LOW = 20
MIDI_HIGH = 108
NUM_PITCH_CLASSES = 12


def get_chroma(roll, length):
    chroma_matrix = torch.zeros((roll.size()[0], NUM_PITCH_CLASSES), device=roll.device)
    for note in range(NUM_PITCH_CLASSES):
        chroma_matrix[:, note] = torch.sum(roll[:, note::NUM_PITCH_CLASSES], axis=1)
    return chroma_matrix


def SSM(sequence):
    chrom = get_chroma(sequence, sequence.size(0))
    chrom = chrom / (chrom.norm(p=2, dim=1, keepdim=True) + 1e-8)
    return torch.matmul(chrom, chrom.T)


def batch_SSM(seq, batch_size):
    SSMs = []
    for i in range(batch_size):
        seq_i = seq[:, i, :]
        if not torch.isfinite(seq_i).all() or seq_i.shape[0] < 2 or seq_i.sum() == 0:
            continue
        SSMs.append(SSM(seq_i))

    if not SSMs:
        raise ValueError("No valid SSMs generated — all samples were skipped")

    return torch.vstack(SSMs)


def make_batches(data, batch_size, piece_size):
    random.shuffle(data)
    batches = []
    if batch_size > 1:
        num_batches = len(data) // batch_size
        for i in range(0, num_batches):
            batch = torch.cat(list(np.array(data)[i * batch_size: (i + 1) * (batch_size)][:, 0])).view(batch_size,
                                                                                                       piece_size, 128)
            batches.append(batch)
    else:
        for i in range(len(data)):
            piece_size = data[i][0].shape[0]
            batch = data[i][0].view(1, piece_size, 128)
            batches.append(batch)
    return batches


def make_variable_size_batches(data, min_batch_size=3, max_batch_size=128):
    data.sort(key=lambda x: x[2], reverse=False)
    batches = []
    i = 0

    while i < len(data):
        this_batch = []
        pieces_this_batch = 0
        current_beats = data[i][2]

        while i < len(data) and data[i][2] == current_beats:
            just_tensor = data[i][0].view(1, data[i][0].shape[0], 128)
            this_batch.append(just_tensor)
            i += 1
            pieces_this_batch += 1

        if pieces_this_batch >= min_batch_size:
            batch = torch.cat(this_batch, dim=0)

            if batch.shape[0] <= max_batch_size:
                batches.append(batch)
            else:
                n_sub_batches = math.ceil(batch.shape[0] / max_batch_size)
                batches.extend(torch.tensor_split(batch, n_sub_batches, dim=0))

    random.shuffle(batches)
    return batches


def topk_sample_one(sequence, k, temperature=1.5, use_softmax=False):
    # Temperature > 1.0 flattens top-k distribution, preventing collapse to 1-2 notes.
    # use_softmax=True avoids sparsemax zeroing candidates when prefix is OOD.
    vals, indices = torch.topk(sequence[:, :, MIDI_LOW:MIDI_HIGH], k)
    indices += MIDI_LOW
    if use_softmax:
        probs = torch.softmax((vals / temperature).float(), dim=2)
    else:
        probs = sparsemax.Sparsemax(dim=2)((vals / temperature).float())
    samples = torch.distributions.Categorical(probs).sample()
    return F.one_hot(torch.gather(indices, -1, samples.unsqueeze(-1)), num_classes=sequence.shape[2]).squeeze(dim=2)


def topk_batch_sample(sequence, k, temperature=1.5, use_softmax=False, n_samples=3):
    acc = None
    for _ in range(n_samples):
        s = topk_sample_one(sequence, k, temperature=temperature, use_softmax=use_softmax)
        acc = s if acc is None else acc + s
    return torch.where(acc > 0, 1, 0)
