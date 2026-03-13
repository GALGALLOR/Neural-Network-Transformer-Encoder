# Transformer Encoder for IMDB Sentiment Classification

This project implements an encoder-only Transformer from scratch in PyTorch for the homework option:

> Option B: Transformer Architecture  
> Implement a Transformer Encoder from scratch for a sequence task, without using `nn.Transformer` or other pre-built Transformer layers.

This project uses the allowed sequence task of text classification on IMDB reviews. It does not do next-token generation. That is intentional and matches the assignment.

## What The Model Predicts

The model predicts sentiment for a whole review:

- `0` = negative
- `1` = positive

This is still prediction. It is just sequence classification rather than sequence-to-sequence or next-token prediction.

## Project Structure

```text
.
|-- main.py
|-- README.md
|-- artifacts/
`-- src
    |-- controllers
    |   |-- __init__.py
    |   `-- train_controller.py
    |-- data
    |   |-- __init__.py
    |   `-- imdb_dataset.py
    |-- models
    |   |-- __init__.py
    |   |-- encoderBlock.py
    |   |-- feedForward.py
    |   |-- models.py
    |   |-- multiHeadSelfAttention.py
    |   |-- positionalEncoding.py
    |   |-- scaledDotProductAttention.py
    |   `-- transformerEncoderClassifier.py
    `-- views
        |-- __init__.py
        `-- cli_view.py
```

## Where The Transformer Is Implemented

The encoder is split across these files:

- `src/models/positionalEncoding.py`
- `src/models/scaledDotProductAttention.py`
- `src/models/multiHeadSelfAttention.py`
- `src/models/feedForward.py`
- `src/models/encoderBlock.py`
- `src/models/transformerEncoderClassifier.py`

The full model class is in `src/models/transformerEncoderClassifier.py`.

## How To Run The Project

You now have two top-level run files:

- `train.py`
- `predict.py`

Edit variables at the top of each file, then run the file directly.

## Train With Hugging Face IMDB

In `train.py`:

```python
TRAINING_CONFIG = TrainingConfig(
    dataset_source="huggingface",
    dataset_name="mteb/imdb",
    data_dir=None,
    batch_size=8,
    epochs=3,
    max_train_samples=500,
    max_test_samples=200,
    checkpoint_path="artifacts/transformer_encoder_classifier.pt",
)
```

Then run:

```powershell
python train.py
```

Install the dependency first if needed:

```powershell
pip install datasets
```

If your environment requires authentication:

```powershell
huggingface-cli login
```

## Train With Local `aclImdb`

In `train.py`:

```python
TRAINING_CONFIG = TrainingConfig(
    dataset_source="local",
    data_dir=r"C:\path\to\aclImdb",
    batch_size=8,
    epochs=3,
    checkpoint_path="artifacts/transformer_encoder_classifier.pt",
)
```

Your local folder should look like:

```text
aclImdb/
|-- train/
|   |-- pos/
|   `-- neg/
`-- test/
    |-- pos/
    `-- neg/
```

## Predict New Text

First train once so the checkpoint exists in `artifacts/`.

Then in `predict.py`:

```python
PREDICTION_TEXT = "This movie was excellent and emotionally engaging."
CHECKPOINT_PATH = "artifacts/transformer_encoder_classifier.pt"
```

Then run:

```powershell
python predict.py
```

The output will show:

- predicted sentiment
- positive probability
- the text you scored

## What `artifacts/` Is

`artifacts/` is an output folder created by training.

It stores generated files such as:

- the saved model checkpoint
- the saved vocabulary inside that checkpoint
- the model hyperparameters inside that checkpoint

This project currently saves:

- `artifacts/transformer_encoder_classifier.pt`

That file now contains:

- model weights
- vocabulary
- model configuration needed for prediction

## What `__init__.py` Does

`__init__.py` marks a folder as a Python package and can re-export useful symbols.

Example:

- `src/models/__init__.py` lets other files write `from src.models import TransformerEncoderClassifier`
- `src/controllers/__init__.py` exposes training and prediction helpers
- `src/views/__init__.py` exposes print helpers

It is mostly package wiring, not where the core logic lives.

## Training Flow

The data flow is:

1. Load raw reviews from local folders or Hugging Face
2. Tokenize text with a simple tokenizer
3. Build a vocabulary from training text
4. Convert tokens to ids
5. Pad/truncate each sequence to `max_length`
6. Create a padding mask
7. Pass ids through token embeddings
8. Add positional encoding
9. Pass through stacked encoder blocks
10. Mean-pool across the sequence
11. Run the classifier layer
12. Compute binary loss with `BCEWithLogitsLoss`

## Why This Matches The Assignment

The assignment says:

- implement a Transformer Encoder from scratch
- for a sequence task such as text classification or time-series

So this project is valid because:

- it manually implements self-attention
- it manually implements positional encoding
- it does not use `nn.Transformer`
- it solves a sequence task: IMDB sentiment classification

If you wanted next-token prediction instead, that would usually require a decoder-style or causal setup, which is a different design from this encoder classifier.
