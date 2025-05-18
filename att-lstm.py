import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
#this package is used to write it back into music.
from mido import Message, MidiFile, MidiTrack
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import random
import torch.distributions
import sparsemax
import math

device = "cuda:0"

exp_path = "/Users/valerizab/Desktop/masters-diploma/sing-learned-structure"
print("loading data from", exp_path + "data/mar-1-variable_bin_bounds_train.csv"+"...")
data = torch.load(exp_path + "data/mar-1-variable_bin_bounds_train.csv")  # truncate rather than padding w/ silence

print("done loading!")
torch.manual_seed(2022)

#this function takes in the piece of music and returns the chroma vectors
def get_chroma(roll, length):
    chroma_matrix = torch.zeros((roll.size()[0],12))
    for note in range(0, 12):
        chroma_matrix[:, note] = torch.sum(roll[:, note::12], axis=1)
    return chroma_matrix

#this takes in the sequence and creates a self-similarity matrix (it calls chroma function inside)
def SSM(sequence):
  #tensor will be in form length, hidden_size (128)
  cos = nn.CosineSimilarity(dim=1)
  chrom = get_chroma(sequence, sequence.size()[0])
  len = chrom.size()[0]
  SSM=torch.zeros((len, len))
  for i in range(0, len):
    SSM[i] = cos(chrom[i].view(1, -1),chrom)
  return (SSM)

#this bundles the SSM function.
def batch_SSM(seq, batch_size):
  # takes sequence in format
  # [beats=400, batch_size, 128]
  # print("SSM\tsequence_shape", seq.shape)
  SSMs = []
  for i in range(0, batch_size):
    # print("SSM\tsequence", seq[:,i,:].shape)
    ssm = SSM(seq[:,i,:])  # [beats, batch, 128]
    # print("SSM\tssm", ssm.shape)
    SSMs.append(ssm)  
  return torch.vstack(SSMs)

# Takes in the batch size and data and returns batches of the batch size
def make_batches(data, batch_size, piece_size):
  random.shuffle(data)
  batches = []
  if batch_size > 1:  # make batches
    print(len(data), batch_size)
    num_batches = len(data)//batch_size
    for i in range(0, num_batches):
      batch = torch.cat(list(np.array(data)[i*batch_size: (i+1)*(batch_size)][:, 0])).view(batch_size, piece_size, 128)
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
  data.sort(key = lambda x: x[2], reverse=False)  # sort descending
  # split data into batches, where each batch contains pieces of the same size
  batches = []

  i = 0  # counter of pieces
  
  while i < len(data):
    this_batch = []
    pieces_this_batch = 0
    current_beats = data[i][2] # num beats in this batch

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

#sampling function 
def topk_sample_one(sequence, k):
  #takes in size sequence length, batch size, values
  softmax = sparsemax.Sparsemax(dim=2)
  vals, indices = torch.topk(sequence[:, :, 20:108],k)
  indices+=20
  seq = torch.distributions.Categorical(softmax(vals.float()))
  samples = seq.sample()
  onehot = F.one_hot(torch.gather(indices, -1, samples.unsqueeze(-1)), num_classes = sequence.shape[2]).squeeze(dim=2)
  return(onehot)

#samples multiple times for the time-step
def topk_batch_sample(sequence, k):
  for i in range(0, 3):
    new= topk_sample_one(sequence, k)
    if i ==0:
      sum = new
    else:
      sum+=new
  return(torch.where(sum>0, 1, 0))

def custom_loss(output, target):
  #custom loss function
  criterion = nn.BCEWithLogitsLoss()
  weighted_mse = criterion(output.double(), target.double())
  batch_size = output.size()[1]
  ssm_err = 0
  for i in range(0, batch_size):
    SSM1 = SSM(output[:,i,:])
    SSM2 = SSM(target[:,i,:])
    ssm_err += (torch.sum((SSM1-SSM2)**2)/(SSM2.size(0)**2))


  return torch.sum(weighted_mse)+ssm_err

# this is the model
class music_generator(nn.Module):
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
        hidden = (torch.zeros(1, batch_size, self.hidden_size)).float().to(device)  # [layers, batch_size, hidden_size/features]
    
        self.hidden = (hidden, hidden) # hidden_state, cell_state
        return

    def set_random_hidden(self, batch_size):
        # create new random hidden layer
        hidden = (torch.randn(1, batch_size, self.hidden_size)).float().to(device)
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
        output, self.hidden = self.lstm(in_put.float().to(device), self.hidden)
        # output dimensions: [sequence_length, batch_size, 128]
        # outputs as many beats (sequence_length) as there were beats in the input
        # hidden: last hidden states from last beat

        #########################
        # attention starts here #
        #########################
        
        # output without attention
        avg_output = output.view(sequence_length, batch_size, 128)  # reshape
        
        # if we're using a starter sequence, cut output to last note
        avg_output = avg_output[-1,:,:].unsqueeze(1)  # [batch_size, 1, 128]
        
        # return early (w/o attention) for base lstm
        if self.base_lstm:
          return avg_output.transpose(0,1), self.hidden
        
        #this variable holds the output after the attention has been applied.
        seqs = []

        # slice the batched ssms to the right places
        beat_num = prev_sequence.shape[0]
        
        # find the row for this beat in each ssm
        # batched_ssm shape is (batch_size*beats, beats), bc all the pieces are stacked vertically atop each other
        inds_across_pieces = range(beat_num, batched_ssm.shape[0], batched_ssm.shape[1])  # eg 11, 2625, 105 - indices of this beat in each of the pieces in the batched_ssm
        # for the row for this beat in each ssm, slice the row up to (not including) this beat
        ssm_slice = batched_ssm[inds_across_pieces, :beat_num] # [batch_size, beat_num]
        # sparsemax makes entries in the vector add to 1
        weights = self.softmax(ssm_slice)  # weights are shape [batch_size, beat_num]

        # this is the sparsemaxed SSM multiplied by the entire previous sequence
        # to scale the previous timesteps for how much attention to pay to each
        # TODO: replace .T
        weighted = (prev_sequence.permute(2,1,0)*weights).T  # [beat_num, batch_size, 128]

        # then it's summed to provide weights for each note.
        weight_vec = (torch.sum(weighted, axis=0)).unsqueeze(1).to(device)  # [batch_size, 1, 128]

        # This concatenates the weights for each note with the output for that note, which is then run through the linear layer to get the final output.
        # returns attentioned note
        pt2 = torch.hstack((weight_vec, avg_output)).transpose(1,2)
        attentioned = self.attention(pt2.float()).permute(2,0,1)  # before .permute() .to("cuda:0")).to('cpu')

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

class model_trainer():
  def __init__(self, generator, optimizer, data, hidden_size=128, batch_size=50):
    self.generator = generator
    self.optimizer = optimizer
    self.batch_size = batch_size  # play with this
    self.hidden_size = hidden_size  # 128
    self.data = data
    self.data_length = data[0][0].shape[0]  # as long as piece length doesn't vary

  def train_epochs(self, num_epochs=50, full_training=False, variable_size_batches=False, save_name="model"):
    #trains each epoch
    losslist = []
    #useful when you want to see the progression of the SSM over time
    piclist = []

    for iter in tqdm(range(0, num_epochs)):
      # start training the generator
      self.generator.train()

      if variable_size_batches:
        # use all data, and group batches by piece size
        batches = make_variable_size_batches(self.data, 1)
      elif full_training and not variable_size_batches: # truncating data doesn't work w/ variable size batches currently
        # use all data
        batches = make_batches(self.data, self.batch_size, self.data_length)
      else:
        # use first 100 pieces
        # can we overfit on a small dataset? if so, can be a good thing b/c shows the model can learn
        batches = make_batches(self.data[:100], self.batch_size, self.data_length)

      cum_loss = 0
      for batch_num in tqdm(range(len(batches))):
        batch = batches[batch_num]
        if full_training:
          # train on full-length pieces
          loss = self.train(batch)
        else:
          # train on first 105 beats of each piece
          loss = self.train(batch[:,:105,:])  # [batch, beats, 128]
        cum_loss+=loss
        del batch
        del loss
      del batches
          
      # print loss for early stopping
      print(cum_loss)
    
      # save generator after each epoch
      curr_file = f"models/{save_name}-epoch-{str(iter)}-loss-{cum_loss:.5f}.txt"
      # !touch curr_file
      torch.save(self.generator, curr_file)

      # generate example piece for piclist
      #snap = self.generate_n_examples(n=1, length=95, starter_notes=10)

      losslist.append(cum_loss) 
      #piclist.append(snap)
        
      # early stopping:
      # after each epoch,
      # run w/ validation
      # if devset (validation) loss goes up for ~5 epochs in a row, early stopping
    return losslist, piclist

  # train for one batch
  def train(self, batch, starter_notes=10):
    # seed vectors for the beginning:
    batch_size = batch.shape[0]
    self_sim = batch_SSM(batch.transpose(0,1), batch_size)  # use variable batch size
    sequence = batch[:,0:starter_notes,:].transpose(0,1)  # start w/ some amount of the piece - 10 might be a bit much
    generated = batch[:,0:starter_notes,:].transpose(0,1)

    # reset hidden to zeros for each batch
    self.generator.init_hidden(batch_size)
        
    # zero the gradients before training for each batch
    self.optimizer.zero_grad()
    
    # for accumulating loss
    loss = 0

    # first .forward on sequence of num_starter_beats (~5 or 10 or so)
    # then loop from there to generate one more element
    next_element = sequence.to("cpu")  # make copy!

    # take
    for i in range(0,batch.shape[1]-starter_notes):  # for each beat
      # iterate through beats, generating for each piece in the batch as you go
      val = torch.rand(1)  # probability it uses original - teacher forcing

      # generate a beat for each piece in the batch
      # we need to do this even in cases of teacher forcing, so we can calculate loss
      output, _ = self.generator.forward(next_element, batch_size, sequence, self_sim)  # returns output, hidden - we don't need the latest copy of hidden
      # print("outside output:", output.shape)
        
      if (val > .8):
        # teacher forcing - 20% of the time,  use original from piece instead of output
        next_element = batch[:,i+1,:].unsqueeze(0)  # [1, 0/deleted, 128] to [1, 1, 128]
      else:
        # 80% of the time we keep the output
        # take last output for each batch
        next_element = topk_batch_sample(output, 50) # sample up to 5 most likely notes at this beat
      
      # add next_element (either generated or teacher) to sequence
      sequence = torch.vstack((sequence, next_element.to("cpu"))) # .unsqueeze(0)
      # append output (generated - not teacher forced) for loss
      generated = torch.vstack((generated, output.to('cpu')))  # used for loss
    
    # run loss after training on whole length of the pieces in the batches
    single_loss = custom_loss(generated[starter_notes:,:,:], batch.transpose(0,1)[starter_notes:,:,:])
    single_loss.backward()

    # update the parameters of the LSTM after running on full batch
    self.optimizer.step()

    loss += single_loss.detach().to('cpu')
    del next_element
    del self_sim
    del sequence
    del generated
    del single_loss
    return (loss)

  def generate_n_pieces(self, initial_vectors, n_pieces, length, batched_ssm):
    # generates a batch of n new pieces of music

    # freeze generator so it doesn't train anymore
    self.generator.eval()  
    # start generator on random hidden states and cell states
    self.generator.set_random_hidden(n_pieces)
  
    # initial vectors in format [batch_size, num_notes=10, 128]
    # change sequence to [10, batch_size, 128]
    sequence = initial_vectors.transpose(0,1)
    next_element = sequence.to("cpu")

    # can't generate more notes than the ssm has entries
    max_notes = batched_ssm.shape[0]-sequence.shape[0]
    
    # generate [length] more beats for the piece
    # or as many beats as available in the ssm
    for i in range(min(length, max_notes)):  # one at a time
      with torch.no_grad():
        # use n_pieces to generate as the batch size
        output, _ = self.generator.forward(next_element.float(), n_pieces, sequence, batched_ssm)
        next_element = topk_batch_sample(output, 50)  # sample up to 5 most likely notes at this beat
      # add element to sequence
      sequence = torch.vstack((sequence, next_element.to("cpu")))

    # return sequence of beats
    return sequence
  

  def generate_n_examples(self, n=1, length=390, starter_notes=10, piece_inds=[0], random_source_pieces=False):
    # get pieces from the data
    pieces = torch.vstack([self.data[i][0].unsqueeze(0) for i in piece_inds]) # get just the note for each piece, and stack pieces
    
    # print(pieces.shape)

    # take first 10 notes in format [1, 10, 128]
    first_vecs = pieces[:,0:starter_notes,:]

    # create batched SSMs for each piece
    batched_ssms = batch_SSM(pieces.transpose(0,1), n)

    # generate pieces
    new_gen = self.generate_n_pieces(first_vecs, n, length, batched_ssms)

    # clean up variables
    del pieces
    del first_vecs
    del batched_ssms

    # return pieces
    return new_gen
print("initializing model...")
# # create model and optimizer
generator = music_generator(128,128, base_lstm=True).to(device)  
optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

# parameters
hidden_size = 128  # maybe don't touch?
print("training for 30 epochs!")
# model trainer
trainer = model_trainer(generator, optimizer, data, hidden_size,)
losslist, piclist = trainer.train_epochs(num_epochs=30,
                                         full_training=True,
                                         variable_size_batches=True
                                        )
torch.save(generator, exp_path + "trained/model_30_epochs.txt" )
plt.scatter(range(0,len(losslist)), losslist)
plt.savefig("/exp/shager/music-gen/plots/train_loss.png")
plt.close()
