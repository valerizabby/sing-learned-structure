# How to prepare LMD data

First of all, we should gather and combine data, to make as it clear as possible. I will use the [LMD dataset](https://colinraffel.com/projects/lmd/), specifically LMD-aligned. 

P.S. Getting a full version requires asking for it: craffel@gmail.com

Secondly, we should get a metadata for this dataset, for example from [hugging face](https://huggingface.co/datasets/ohollo/lmd_chords/blob/4d6815cdd528bd1e99dcdefcb06d6f40429ec128/README.md).

We need to combine these sources: for each `track_id` find paths to MIDI and Audio files and store them in Metadata in Maestro-like style.
Building maestro-like dataset for LMD stored in `build_LMD.py`

## Data preparation pipeline

1) Run `build_LMD.py` (change LMD_ROOT, LMD_AUDIO_ROOT);
2) Run `data_preprocessing.py` (change INPUT_PARQUET, OUTPUT_PICKLE);
3) Run `build_BINS.py` to build train, val and test sets for 
