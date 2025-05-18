import torch
from tqdm import tqdm

from SingLS.data_utils import batch_SSM, topk_batch_sample, custom_loss, make_variable_size_batches, make_batches


class ModelTrainer:
    def __init__(self, generator, optimizer, data, hidden_size=128, batch_size=50):
        self.generator = generator
        self.optimizer = optimizer
        self.batch_size = batch_size  # play with this
        self.hidden_size = hidden_size  # 128
        self.data = data
        self.data_length = data[0][0].shape[0]  # as long as piece length doesn't vary

    def train_epochs(self, num_epochs=50, full_training=False, variable_size_batches=False, save_name="model"):
        # trains each epoch
        losslist = []
        # useful when you want to see the progression of the SSM over time
        piclist = []

        for iter in tqdm(range(0, num_epochs)):
            # start training the generator
            self.generator.train()

            if variable_size_batches:
                # use all data, and group batches by piece size
                batches = make_variable_size_batches(self.data, 1)
            elif full_training and not variable_size_batches:  # truncating data doesn't work w/ variable size batches currently
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
                    loss = self.train(batch[:, :105, :])  # [batch, beats, 128]
                cum_loss += loss
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
            # snap = self.generate_n_examples(n=1, length=95, starter_notes=10)

            losslist.append(cum_loss)
            # piclist.append(snap)

            # early stopping:
            # after each epoch,
            # run w/ validation
            # if devset (validation) loss goes up for ~5 epochs in a row, early stopping
        return losslist, piclist

    # train for one batch
    def train(self, batch, starter_notes=10):
        # seed vectors for the beginning:
        batch_size = batch.shape[0]
        self_sim = batch_SSM(batch.transpose(0, 1), batch_size)  # use variable batch size
        sequence = batch[:, 0:starter_notes, :].transpose(0,
                                                          1)  # start w/ some amount of the piece - 10 might be a bit much
        generated = batch[:, 0:starter_notes, :].transpose(0, 1)

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
        for i in range(0, batch.shape[1] - starter_notes):  # for each beat
            # iterate through beats, generating for each piece in the batch as you go
            val = torch.rand(1)  # probability it uses original - teacher forcing

            # generate a beat for each piece in the batch
            # we need to do this even in cases of teacher forcing, so we can calculate loss
            output, _ = self.generator.forward(next_element, batch_size, sequence,
                                               self_sim)  # returns output, hidden - we don't need the latest copy of hidden
            # print("outside output:", output.shape)

            if (val > .8):
                # teacher forcing - 20% of the time,  use original from piece instead of output
                next_element = batch[:, i + 1, :].unsqueeze(0)  # [1, 0/deleted, 128] to [1, 1, 128]
            else:
                # 80% of the time we keep the output
                # take last output for each batch
                next_element = topk_batch_sample(output, 50)  # sample up to 5 most likely notes at this beat

            # add next_element (either generated or teacher) to sequence
            sequence = torch.vstack((sequence, next_element.to("cpu")))  # .unsqueeze(0)
            # append output (generated - not teacher forced) for loss
            generated = torch.vstack((generated, output.to('cpu')))  # used for loss

        # run loss after training on whole length of the pieces in the batches
        single_loss = custom_loss(generated[starter_notes:, :, :], batch.transpose(0, 1)[starter_notes:, :, :])
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
        sequence = initial_vectors.transpose(0, 1)
        next_element = sequence.to("cpu")

        # can't generate more notes than the ssm has entries
        max_notes = batched_ssm.shape[0] - sequence.shape[0]

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
        pieces = torch.vstack(
            [self.data[i][0].unsqueeze(0) for i in piece_inds])  # get just the note for each piece, and stack pieces

        # print(pieces.shape)

        # take first 10 notes in format [1, 10, 128]
        first_vecs = pieces[:, 0:starter_notes, :]

        # create batched SSMs for each piece
        batched_ssms = batch_SSM(pieces.transpose(0, 1), n)

        # generate pieces
        new_gen = self.generate_n_pieces(first_vecs, n, length, batched_ssms)

        # clean up variables
        del pieces
        del first_vecs
        del batched_ssms

        # return pieces
        return new_gen
