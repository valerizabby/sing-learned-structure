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

**data_processing.ipynb**, as the name suggests, processes the MAESTRO dataset into a form usable by SING. This repository already contains preprocessed data, so it is unnecessary to rerun this code.

**att_lstm.py** is the file used to train SING and the LSTM ablation. Running it will train the model again; by default, it will train for 30 epochs. The "best_models" folder contains the best models with and without the attention mechanism.

**model-selection.py** can be used to find the model with lowest loss on the validation dataset.

**test-sim.ipynb** contains the code used to find MSE over the standardized SSM.

**gen-example.ipynb** uses the best pre-trained model to generate new pieces of music. From the notebook, you can both view the self-similarity matrix of the new generated piece and save the audio of the piece to "blank.midi", a file that is overwritten each time the relevant code in the notebook is run.

These notebooks were initially designed to be run on Google Colab, and adapting to run locally will require edits to the code.