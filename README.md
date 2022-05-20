# SING
Similarity Incentivized Neural Generator, or SING, is a music generation system which uses self-similarity as attention.

## Required packages

- numpy
- pandas
- matplotlib
- pretty_midi
- mido
- PyTorch 
- tqdm 
- sparsemax

## Running the code

Each notebook has its own purpose.

**data_processing.ipynb**, as the name suggests, processes the MAESTRO data into a form usable by SING. This repository already contains preprocessed data in train_tempo.csv, test_tempo.csv, and validation_tempo.csv, so it is unnecessary to rerun this code for the other files to function.

**att_lstm.ipynb** is the file used to train SING. Running it will train the model again; by default, it will train for 30 epochs. The "trained" folder contains pre-trained examples.

**generation.ipynb** uses the best pre-trained model to generate new pieces of music. From the notebook, you can both view the self-similarity matrix of the new generated piece and save the audio of the piece to "blank.midi", a file that is overwritten each time the relevant code in the notebook is run.
