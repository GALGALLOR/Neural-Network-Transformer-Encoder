import torch

from src.controllers import TrainingConfig, run_training
from src.views import print_project_overview


# Edit these values directly before running `python train.py`.
TRAINING_CONFIG = TrainingConfig(
    dataset_source="huggingface",
    dataset_name="mteb/imdb",
    data_dir=None,
    batch_size=16,
    max_length=256,
    max_vocab_size=20000,
    min_freq=2,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=256,
    dropout=0.1,
    learning_rate=1e-4,
    epochs=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_train_samples=100000,
    max_test_samples=20000,
    #max_train_samples=500,
    #max_test_samples=200,
    checkpoint_path="artifacts/transformer_encoder_classifier.pt",
)


def main() -> None:
    print_project_overview()
    run_training(TRAINING_CONFIG)


if __name__ == "__main__":
    main()
