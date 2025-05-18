import torch
import torch.nn as nn
import sparsemax
from config import DEVICE


class MusicGenerator(nn.Module):
    def __init__(self, hidden_size, output_size, base_lstm=False):
        super().__init__()
        self.hidden_size = hidden_size  # 128
        # output_size is num expected features (128)
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers=1, bidirectional=False)
        self.attention = nn.Linear(2, 1)
        self.softmax = sparsemax.Sparsemax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None
        self.base_lstm = base_lstm  # true to use lstm without attention

    def init_hidden(self, batch_size):
        # set hidden state to zeros after each batch
        hidden = (torch.zeros(1, batch_size, self.hidden_size)).float().to(
            DEVICE)  # [layers, batch_size, hidden_size/features]

        self.hidden = (hidden, hidden)  # hidden_state, cell_state
        return

    def set_random_hidden(self, batch_size):
        # create new random hidden layer
        hidden = (torch.randn(1, batch_size, self.hidden_size)).float().to(DEVICE)
        self.hidden = (hidden, hidden)
        return

    def forward(self, in_put, batch_size, prev_sequence, batched_ssm):
        # look at tensor things - view vs. reshape vs. permute, and unsqueeze and squeeze
        # try looking at the LSTM equations
        # .to('cpu')  # returns a copy of the tensor in CPU memory
        # .to('cuda:0')  # returns copy in CUDA memory, 0 indicates first GPU device
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to

        # sequence length
        # size of input (10 or 1)
        sequence_length = in_put.size()[0]
        # print("in_put:", in_put.shape)

        # Run the LSTM
        # output - sequence of all the hidden states
        # hidden - most recent hidden state
        # input dimensions: [sequence_length, batch_size, 128]
        output, self.hidden = self.lstm(in_put.float().to(DEVICE), self.hidden)
        # output dimensions: [sequence_length, batch_size, 128]
        # outputs as many beats (sequence_length) as there were beats in the input
        # hidden: last hidden states from last beat

        #########################
        # attention starts here #
        #########################

        # output without attention
        avg_output = output.view(sequence_length, batch_size, 128)  # reshape

        # if we're using a starter sequence, cut output to last note
        avg_output = avg_output[-1, :, :].unsqueeze(1)  # [batch_size, 1, 128]

        # return early (w/o attention) for base lstm
        if self.base_lstm:
            return avg_output.transpose(0, 1), self.hidden

        # this variable holds the output after the attention has been applied.
        seqs = []

        # slice the batched ssms to the right places
        beat_num = prev_sequence.shape[0]

        # find the row for this beat in each ssm
        # batched_ssm shape is (batch_size*beats, beats), bc all the pieces are stacked vertically atop each other
        inds_across_pieces = range(beat_num, batched_ssm.shape[0], batched_ssm.shape[
            1])  # eg 11, 2625, 105 - indices of this beat in each of the pieces in the batched_ssm
        # for the row for this beat in each ssm, slice the row up to (not including) this beat
        ssm_slice = batched_ssm[inds_across_pieces, :beat_num]  # [batch_size, beat_num]
        # sparsemax makes entries in the vector add to 1
        weights = self.softmax(ssm_slice)  # weights are shape [batch_size, beat_num]

        # this is the sparsemaxed SSM multiplied by the entire previous sequence
        # to scale the previous timesteps for how much attention to pay to each
        # TODO: replace .T
        weighted = (prev_sequence.permute(2, 1, 0) * weights).T  # [beat_num, batch_size, 128]

        # then it's summed to provide weights for each note.
        weight_vec = (torch.sum(weighted, axis=0)).unsqueeze(1).to(DEVICE)  # [batch_size, 1, 128]

        # This concatenates the weights for each note with the output for that note, which is then run through the linear layer to get the final output.
        # returns attentioned note
        pt2 = torch.hstack((weight_vec, avg_output)).transpose(1, 2)
        attentioned = self.attention(pt2.float()).permute(2, 0, 1)  # before .permute() .to("cuda:0")).to('cpu')

        # delete vars to remove clutter in memory
        del pt2
        del weight_vec
        del weighted
        del weights
        del ssm_slice
        del inds_across_pieces
        del beat_num
        del avg_output

        # return attentioned note
        return attentioned.double(), self.hidden  # hidden = hidden_state, cell_state