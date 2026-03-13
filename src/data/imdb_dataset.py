from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


def simple_tokenizer(text: str) -> list[str]:
    return text.lower().split()


@dataclass(frozen=True)
class Vocabulary:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @property
    def pad_idx(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_idx(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(token, self.unk_idx) for token in tokens]

    def __len__(self) -> int:
        return len(self.itos)

    def to_dict(self) -> dict[str, object]:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Vocabulary":
        return cls(
            stoi=dict(payload["stoi"]),
            itos=list(payload["itos"]),
            pad_token=str(payload["pad_token"]),
            unk_token=str(payload["unk_token"]),
        )


def build_vocabulary(
    texts: list[str],
    tokenizer=simple_tokenizer,
    max_size: int = 20000,
    min_freq: int = 2,
) -> Vocabulary:
    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))

    specials = ["<pad>", "<unk>"]
    stoi = {token: index for index, token in enumerate(specials)}
    itos = list(specials)

    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in stoi:
            continue
        if len(itos) >= max_size:
            break
        stoi[token] = len(itos)
        itos.append(token)

    return Vocabulary(stoi=stoi, itos=itos)


class IMDBReviewDataset(Dataset):
    """Reads raw IMDB reviews stored like aclImdb/train/pos and aclImdb/train/neg."""

    def __init__(self, root_dir: str | Path, split: str = "train") -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.samples = self._collect_samples()

    def _resolve_split_dir(self) -> Path:
        direct_split_dir = self.root_dir / self.split
        if direct_split_dir.exists():
            return direct_split_dir

        nested_split_dir = self.root_dir / "aclImdb" / self.split
        if nested_split_dir.exists():
            return nested_split_dir

        raise FileNotFoundError(
            "Could not find the IMDB split directory.\n"
            f"Received --data-dir: {self.root_dir}\n"
            f"Looked for: {direct_split_dir}\n"
            f"Also looked for: {nested_split_dir}\n"
            "Expected folders like train/pos, train/neg, test/pos, and test/neg.\n"
            "Example: python main.py train --data-dir C:\\path\\to\\aclImdb --epochs 3"
        )

    def _collect_samples(self) -> list[tuple[Path, int]]:
        split_dir = self._resolve_split_dir()

        samples: list[tuple[Path, int]] = []
        for label_name, label in (("neg", 0), ("pos", 1)):
            label_dir = split_dir / label_name
            if not label_dir.exists():
                raise FileNotFoundError(
                    f"Missing label directory: {label_dir}\n"
                    "Expected both 'pos' and 'neg' folders inside each split directory."
                )
            for path in sorted(label_dir.glob("*.txt")):
                samples.append((path, label))

        if not samples:
            raise FileNotFoundError(
                f"No .txt review files were found under {split_dir}.\n"
                "Verify that the IMDB dataset was extracted correctly."
            )
        return samples

    def texts(self) -> list[str]:
        return [path.read_text(encoding="utf-8") for path, _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        path, label = self.samples[index]
        text = path.read_text(encoding="utf-8")
        return text, label


class TextLabelDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]]) -> None:
        self.samples = samples

    def texts(self) -> list[str]:
        return [text for text, _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.samples[index]


def _resolve_hf_columns(split) -> tuple[str, str]:
    text_candidates = ("text", "review", "content", "sentence")
    label_candidates = ("label", "sentiment", "target")

    text_column = next((column for column in text_candidates if column in split.column_names), None)
    label_column = next((column for column in label_candidates if column in split.column_names), None)

    if text_column is None or label_column is None:
        raise ValueError(
            f"Unsupported Hugging Face dataset schema. Found columns: {split.column_names}. "
            "Expected a text column like 'text' and a label column like 'label'."
        )

    return text_column, label_column


def load_huggingface_imdb_datasets(
    dataset_name: str = "mteb/imdb",
    train_split: str = "train",
    test_split: str = "test",
) -> tuple[TextLabelDataset, TextLabelDataset]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for Hugging Face loading. "
            "Install it with: pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name)
    if train_split not in dataset or test_split not in dataset:
        raise ValueError(
            f"Dataset '{dataset_name}' does not expose the requested splits "
            f"('{train_split}', '{test_split}'). Available splits: {list(dataset.keys())}"
        )

    train_hf_split = dataset[train_split]
    test_hf_split = dataset[test_split]
    text_column, label_column = _resolve_hf_columns(train_hf_split)

    def convert_split(split) -> TextLabelDataset:
        samples: list[tuple[str, int]] = []
        for row in split:
            samples.append((str(row[text_column]), int(row[label_column])))
        return TextLabelDataset(samples)

    return convert_split(train_hf_split), convert_split(test_hf_split)


def collate_imdb_batch(
    batch: list[tuple[str, int]],
    vocab: Vocabulary,
    max_length: int = 256,
    tokenizer=simple_tokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids: list[list[int]] = []
    labels: list[int] = []

    for text, label in batch:
        token_ids = vocab.encode(tokenizer(text))[:max_length]
        padded = token_ids + [vocab.pad_idx] * max(0, max_length - len(token_ids))
        input_ids.append(padded[:max_length])
        labels.append(label)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )
