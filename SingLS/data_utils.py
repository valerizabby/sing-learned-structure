import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.distributions
import sparsemax
import math

# this function takes in the piece of music and returns the chroma vectors
def get_chroma(roll, length):
    chroma_matrix = torch.zeros((roll.size()[0], 12))
    for note in range(0, 12):
        chroma_matrix[:, note] = torch.sum(roll[:, note::12], axis=1)
    return chroma_matrix


# this takes in the sequence and creates a self-similarity matrix (it calls chroma function inside)
def SSM(sequence):
    # tensor will be in form length, hidden_size (128)
    cos = nn.CosineSimilarity(dim=1)
    chrom = get_chroma(sequence, sequence.size()[0])
    len = chrom.size()[0]
    SSM = torch.zeros((len, len))
    for i in range(0, len):
        SSM[i] = cos(chrom[i].view(1, -1), chrom)
    return (SSM)


# this bundles the SSM function.
def batch_SSM(seq, batch_size):
    # takes sequence in format
    # [beats=400, batch_size, 128]
    # print("SSM\tsequence_shape", seq.shape)
    SSMs = []
    for i in range(0, batch_size):
        # print("SSM\tsequence", seq[:,i,:].shape)
        ssm = SSM(seq[:, i, :])  # [beats, batch, 128]
        # print("SSM\tssm", ssm.shape)
        SSMs.append(ssm)
    return torch.vstack(SSMs)


# Takes in the batch size and data and returns batches of the batch size
def make_batches(data, batch_size, piece_size):
    random.shuffle(data)
    batches = []
    if batch_size > 1:  # make batches
        print(len(data), batch_size)
        num_batches = len(data) // batch_size
        for i in range(0, num_batches):
            batch = torch.cat(list(np.array(data)[i * batch_size: (i + 1) * (batch_size)][:, 0])).view(batch_size,
                                                                                                       piece_size, 128)
            batches.append(batch)
    else:  # each piece is its own batch - doesn't use passed-in piece_size
        for i in range(len(data)):
            # removes tempo info from data, but leaves 1 piece per batch
            piece_size = data[i][0].shape[0]
            batch = data[i][0].view(1, piece_size, 128)
            batches.append(batch)
            # print(batches[i])
    # print(batches)
    return batches


# returns batches where piece size is constant within the batch
# but piece size is different across batches
# and batches are in random order
def make_variable_size_batches(data, min_batch_size=3, max_batch_size=128):
    # sort data by num beats (element at index 2 in each sublist)
    data.sort(key=lambda x: x[2], reverse=False)  # sort descending
    # split data into batches, where each batch contains pieces of the same size
    batches = []

    i = 0  # counter of pieces

    while i < len(data):
        this_batch = []
        pieces_this_batch = 0
        current_beats = data[i][2]  # num beats in this batch

        # for all pieces with this # of beats
        while i < len(data) and data[i][2] == current_beats:
            # get tensor from row of data, and reshape
            just_tensor = data[i][0].view(1, data[i][0].shape[0], 128)
            this_batch.append(just_tensor)

            # increment counters
            i += 1
            pieces_this_batch += 1

        # only save large enough batches
        if pieces_this_batch >= min_batch_size:
            # reformat pieces in this batch into one tensor of size [batch size, beats, 128]
            batch = torch.cat(this_batch, dim=0)

            # if batch exceeds max batch size, split into sub_batches
            if batch.shape[0] <= max_batch_size:
                # store batch
                batches.append(batch)
            else:
                # split batch into equal-size chunks, less than max batch size
                # example, 103 pieces into max size 50 requires 3 splits of equal size 35/34/34
                n_sub_batches = math.ceil(batch.shape[0] / max_batch_size)  # how many chunks are needed
                sub_batches = torch.tensor_split(batch, n_sub_batches, dim=0)

                # store batches
                batches.extend(sub_batches)

        # clean up variables
        del this_batch
        del pieces_this_batch
        del current_beats

    # randomize batches order
    random.shuffle(batches)

    return batches


# sampling function
def topk_sample_one(sequence, k):
    # takes in size sequence length, batch size, values
    softmax = sparsemax.Sparsemax(dim=2)
    vals, indices = torch.topk(sequence[:, :, 20:108], k)
    indices += 20
    seq = torch.distributions.Categorical(softmax(vals.float()))
    samples = seq.sample()
    onehot = F.one_hot(torch.gather(indices, -1, samples.unsqueeze(-1)), num_classes=sequence.shape[2]).squeeze(dim=2)
    return (onehot)


# samples multiple times for the time-step
def topk_batch_sample(sequence, k):
    for i in range(0, 3):
        new = topk_sample_one(sequence, k)
        if i == 0:
            sum = new
        else:
            sum += new
    return (torch.where(sum > 0, 1, 0))


def custom_loss(output, target):
    # custom loss function
    criterion = nn.BCEWithLogitsLoss()
    weighted_mse = criterion(output.double(), target.double())
    batch_size = output.size()[1]
    ssm_err = 0
    for i in range(0, batch_size):
        SSM1 = SSM(output[:, i, :])
        SSM2 = SSM(target[:, i, :])
        ssm_err += (torch.sum((SSM1 - SSM2) ** 2) / (SSM2.size(0) ** 2))

    return torch.sum(weighted_mse) + ssm_err