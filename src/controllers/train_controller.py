from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data import (
    IMDBReviewDataset,
    Vocabulary,
    build_vocabulary,
    collate_imdb_batch,
    load_huggingface_imdb_datasets,
    simple_tokenizer,
)
from src.models import TransformerEncoderClassifier, create_padding_mask


@dataclass
class TrainingConfig:
    data_dir: str | None = None
    dataset_source: str = "local"
    dataset_name: str = "mteb/imdb"
    batch_size: int = 32
    max_length: int = 256
    max_vocab_size: int = 20000
    min_freq: int = 2
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 3
    device: str = "cpu"
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    checkpoint_path: str = "artifacts/transformer_encoder_classifier.pt"


def build_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader, Vocabulary]:
    if config.dataset_source == "huggingface":
        train_dataset, test_dataset = load_huggingface_imdb_datasets(config.dataset_name)
    else:
        if not config.data_dir:
            raise ValueError("--data-dir is required when --dataset-source local is used.")
        train_dataset = IMDBReviewDataset(config.data_dir, split="train")
        test_dataset = IMDBReviewDataset(config.data_dir, split="test")

    if config.max_train_samples is not None:
        train_dataset = Subset(
            train_dataset,
            range(min(config.max_train_samples, len(train_dataset))),
        )
    if config.max_test_samples is not None:
        test_dataset = Subset(
            test_dataset,
            range(min(config.max_test_samples, len(test_dataset))),
        )

    vocab = build_vocabulary(
        [text for text, _ in train_dataset],
        max_size=config.max_vocab_size,
        min_freq=config.min_freq,
    )

    collate_fn = partial(collate_imdb_batch, vocab=vocab, max_length=config.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader, vocab


def create_model(config: TrainingConfig, vocab_size: int) -> TransformerEncoderClassifier:
    return TransformerEncoderClassifier(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_len=config.max_length,
        dropout=config.dropout,
    )


def _model_config_payload(config: TrainingConfig, vocab_size: int) -> dict[str, int | float]:
    return {
        "vocab_size": vocab_size,
        "d_model": config.d_model,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "d_ff": config.d_ff,
        "max_len": config.max_length,
        "dropout": config.dropout,
    }


def save_checkpoint(
    model: TransformerEncoderClassifier,
    vocab: Vocabulary,
    config: TrainingConfig,
) -> Path:
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab.to_dict(),
            "model_config": _model_config_payload(config, len(vocab)),
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[TransformerEncoderClassifier, Vocabulary, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = Vocabulary.from_dict(checkpoint["vocab"])
    model_config = checkpoint["model_config"]
    model = TransformerEncoderClassifier(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab, checkpoint


def predict_text(
    text: str,
    checkpoint_path: str,
    device: str = "cpu",
) -> dict[str, object]:
    torch_device = torch.device(device)
    model, vocab, checkpoint = load_checkpoint(checkpoint_path, torch_device)
    max_length = int(checkpoint["model_config"]["max_len"])

    token_ids = vocab.encode(simple_tokenizer(text))[:max_length]
    padded = token_ids + [vocab.pad_idx] * max(0, max_length - len(token_ids))
    input_ids = torch.tensor([padded[:max_length]], dtype=torch.long, device=torch_device)
    mask = create_padding_mask(input_ids).to(torch_device)

    with torch.no_grad():
        logits, _ = model(input_ids, mask)
        probability = torch.sigmoid(logits.squeeze()).item()

    predicted_label = 1 if probability >= 0.5 else 0
    predicted_sentiment = "positive" if predicted_label == 1 else "negative"
    return {
        "text": text,
        "probability_positive": probability,
        "predicted_label": predicted_label,
        "predicted_sentiment": predicted_sentiment,
    }


def train_one_epoch(
    model: TransformerEncoderClassifier,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for input_ids, labels in data_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        mask = create_padding_mask(input_ids).to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids, mask)
        loss = criterion(logits.squeeze(1), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(data_loader))


def evaluate(
    model: TransformerEncoderClassifier,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            mask = create_padding_mask(input_ids).to(device)

            logits, _ = model(input_ids, mask)
            loss = criterion(logits.squeeze(1), labels)

            probs = torch.sigmoid(logits.squeeze(1))
            preds = (probs >= 0.5).float()

            total_loss += loss.item()
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

    avg_loss = total_loss / max(1, len(data_loader))
    accuracy = total_correct / max(1, total_examples)
    return avg_loss, accuracy


def run_training(config: TrainingConfig) -> dict[str, float]:
    train_loader, test_loader, vocab = build_dataloaders(config)
    device = torch.device(config.device)
    print(
        f"Starting training | source={config.dataset_source} | device={device} | "
        f"train_batches={len(train_loader)} | test_batches={len(test_loader)} | vocab_size={len(vocab)}"
    )

    model = create_model(config, len(vocab)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: dict[str, float] = {}
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)
        history = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f}"
        )

    checkpoint_path = save_checkpoint(model, vocab, config)
    print(f"Saved checkpoint: {checkpoint_path}")
    return history


def run_smoke_test() -> tuple[torch.Size, int]:
    batch_size = 4
    seq_len = 10
    vocab_size = 1000

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = create_padding_mask(input_ids)

    model = TransformerEncoderClassifier(vocab_size=vocab_size)
    logits, attention_weights = model(input_ids, mask)
    return logits.shape, len(attention_weights)
