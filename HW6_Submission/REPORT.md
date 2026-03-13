# Homework 6 Report

## Student Work Summary

This submission contains two parts:

1. Foundation Module: Fashion-MNIST image classification using a convolutional neural network.
2. Advanced Track Option B: A Transformer Encoder implemented from scratch for IMDB sentiment classification.

The overall goal of this homework was to implement nontrivial neural architectures and document the design choices, training setup, and results. For the foundation task, I built and trained a CNN on Fashion-MNIST. For the advanced track, I implemented the attention mechanism, positional encoding, and stacked encoder blocks manually in PyTorch without using `nn.Transformer`.

## Part 1: Foundation Module

### Task

The foundation task was Fashion-MNIST classification. The objective was to classify grayscale clothing images into one of ten categories. The assignment allowed either an MLP or a simple CNN. I selected a convolutional neural network because CNNs are better suited to spatial image data and generally outperform fully connected models on this dataset.

### Data Preparation

The Fashion-MNIST dataset was loaded through `torchvision.datasets.FashionMNIST`. I used separate transforms for training and evaluation:

- Training transform:
  - random rotation of 5 degrees
  - conversion to tensor
  - normalization with mean `0.5` and standard deviation `0.5`
- Validation/test transform:
  - conversion to tensor
  - normalization with mean `0.5` and standard deviation `0.5`

The original 60,000 training samples were split into:

- 50,000 training samples
- 10,000 validation samples

The test split remained the standard 10,000 images. Data was loaded with a batch size of `128`.

### CNN Architecture

The model used a three-stage convolutional feature extractor followed by a fully connected classifier.

Convolutional portion:

- Block 1:
  - `Conv2d(1, 64, kernel_size=3, padding=1)`
  - BatchNorm
  - ReLU
  - `Conv2d(64, 64, kernel_size=3, padding=1)`
  - BatchNorm
  - ReLU
  - MaxPool
  - Dropout `0.10`
- Block 2:
  - `Conv2d(64, 128, kernel_size=3, padding=1)`
  - BatchNorm
  - ReLU
  - `Conv2d(128, 128, kernel_size=3, padding=1)`
  - BatchNorm
  - ReLU
  - MaxPool
  - Dropout `0.15`
- Block 3:
  - `Conv2d(128, 256, kernel_size=3, padding=1)`
  - BatchNorm
  - ReLU
  - `Conv2d(256, 256, kernel_size=3, padding=1)`
  - BatchNorm
  - ReLU
  - MaxPool
  - Dropout `0.20`

Fully connected portion:

- `Linear(256 * 3 * 3, 256)`
- ReLU
- Dropout `0.30`
- `Linear(256, 128)`
- ReLU
- Dropout `0.20`
- `Linear(128, 10)`

The model was trained with:

- loss: `CrossEntropyLoss`
- optimizer: `Adam`
- learning rate: `0.0003`
- weight decay: `1e-4`
- scheduler: `ReduceLROnPlateau`

### Results

The model trained for 50 epochs and the best validation accuracy during training reached `0.9470` at epoch 22. After restoring the best validation checkpoint, the final test result was:

- Test Loss: `0.2283`
- Test Accuracy: `0.9405`

These results show that the CNN learned useful visual features and generalized reasonably well, but the current run did not reach the assignment target of greater than 98% test accuracy. The training and validation curves suggest that the model improved steadily early on and then plateaued, with some overfitting in later epochs as training accuracy continued rising while validation accuracy stabilized.

### Interpretation

The Fashion-MNIST model is a competent CNN baseline, but it would likely need additional tuning to satisfy the stated threshold. Potential improvements include stronger augmentation, a revised learning-rate schedule, more careful regularization, or architecture changes such as residual connections or a different capacity profile.

## Part 2: Advanced Track Option B

### Task

For the advanced track, I implemented a Transformer Encoder from scratch for a sequence classification task using IMDB movie reviews. The objective was binary sentiment prediction: classify each review as positive or negative.

The main constraint was that pre-built Transformer layers could not be used. This means the self-attention mechanism and positional encoding had to be implemented manually rather than relying on `nn.Transformer`.

### Model Design

I implemented the Transformer as an encoder-only architecture with the following sequence of operations:

`input_ids -> token embedding -> positional encoding -> encoder blocks -> mean pooling -> classifier`

The implementation was separated into modular components:

- `PositionalEncoding`
- `ScaledDotProductAttention`
- `MultiHeadSelfAttention`
- `FeedForward`
- `EncoderBlock`
- `TransformerEncoderClassifier`

#### Positional Encoding

The positional encoding was sinusoidal and precomputed once up to a maximum sequence length. The encoding was stored as a non-trainable buffer and added to the token embeddings during the forward pass.

#### Scaled Dot-Product Attention

Attention was implemented directly from the formula:

`softmax(QK^T / sqrt(d_k))V`

This included:

- projection into query, key, and value spaces
- transpose of keys for score computation
- scaling by the square root of the head dimension
- masking of padding tokens
- softmax normalization over token positions
- multiplication by the value tensor

#### Multi-Head Self-Attention

The encoder uses multiple attention heads. The input tensor is linearly projected to Q, K, and V, reshaped into `(batch, heads, sequence, head_dim)`, processed by attention, then recombined into the original model dimension.

#### Encoder Block

Each encoder block contains:

- multi-head self-attention
- residual connection
- layer normalization
- feedforward network
- residual connection
- layer normalization

This matches the standard Transformer encoder design.

### Training Setup

The model was trained on IMDB reviews loaded through Hugging Face `datasets`, using `mteb/imdb`. The sequence task was binary classification, so the final output layer had one logit and training used `BCEWithLogitsLoss`.

The main training configuration was:

- `d_model = 128`
- `num_heads = 4`
- `num_layers = 2`
- `d_ff = 256`
- `max_length = 256`
- dropout `0.1`
- optimizer: Adam
- learning rate: `1e-4`

The project also included utilities for:

- building a vocabulary from training text
- converting text to token ids
- padding and masking variable-length sequences
- saving a checkpoint that includes both model weights and vocabulary
- running separate training and prediction scripts

### Results

A representative full training run produced the following validation history:

- Epoch 1: `train_loss=0.6027`, `val_loss=0.4743`, `val_acc=0.7700`
- Epoch 5: `train_loss=0.3766`, `val_loss=0.3836`, `val_acc=0.8252`
- Epoch 10: `train_loss=0.2867`, `val_loss=0.3704`, `val_acc=0.8424`
- Epoch 14: `train_loss=0.2332`, `val_loss=0.3907`, `val_acc=0.8486`
- Epoch 20: `train_loss=0.1622`, `val_loss=0.4468`, `val_acc=0.8470`

These results indicate that the model learned meaningful sentiment structure from the text and reached a best validation accuracy of approximately `84.86%`. The training loss decreased steadily across epochs, while the validation accuracy improved early and then plateaued. Validation loss became noisier later in training, which suggests moderate overfitting after the best epoch.

### Interpretation

The Transformer model is structurally correct and solves the assignment as stated:

- it is an encoder-only Transformer
- self-attention was coded manually
- positional encoding was coded manually
- no pre-built Transformer encoder layers were used
- it was applied to a sequence task, namely IMDB sentiment classification

The final results are realistic for a from-scratch implementation with a simple tokenizer. The project also includes prediction support using a saved checkpoint, which allows new review text to be classified after training.

## Discussion

The two parts of the homework emphasize different strengths of deep learning architectures.

The CNN in the foundation module is specialized for image data, where local spatial patterns matter. Convolutions, pooling, and batch normalization help the model learn edges, textures, and progressively more complex visual features.

The Transformer in the advanced module is specialized for sequence modeling. Instead of local spatial filters, it uses self-attention so each token can attend to every other token in the review. Positional encoding provides token order information that the attention mechanism otherwise lacks.

One of the most valuable parts of the advanced track was implementing the internal pieces directly rather than calling a built-in Transformer layer. Doing so made the tensor shapes, masking, residual flow, and attention computation much more explicit.

## Conclusion

In this homework, I completed a CNN-based Fashion-MNIST classifier for the mandatory foundation module and implemented a Transformer Encoder from scratch for the advanced sequence modeling track.

For Fashion-MNIST, the CNN achieved `94.05%` test accuracy, demonstrating strong learning performance but falling short of the assignment's stated `>98%` target. For the advanced track, the Transformer Encoder achieved roughly `84.9%` validation accuracy on IMDB sentiment classification while satisfying the manual-attention and manual-positional-encoding constraints.

Overall, the work demonstrates understanding of both convolutional image models and Transformer-based sequence models, along with the engineering needed to train, evaluate, and save neural architectures in PyTorch.
