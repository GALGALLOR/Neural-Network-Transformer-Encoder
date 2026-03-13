import torch

from src.controllers import TrainingConfig, predict_text, run_smoke_test, run_training
from src.views import print_prediction_result, print_project_overview, print_smoke_test_result


# Edit these values directly before running `python main.py`.
RUN_MODE = "train"
# Options: "smoke_test", "train", "predict"

PREDICTION_TEXT = (
    "This movie was surprisingly good, the acting was strong and the ending worked."
)

TRAINING_CONFIG = TrainingConfig(
    dataset_source="huggingface",
    dataset_name="mteb/imdb",
    data_dir=None,
    batch_size=8,
    max_length=256,
    max_vocab_size=20000,
    min_freq=2,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=256,
    dropout=0.1,
    learning_rate=1e-3,
    epochs=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    #max_train_samples=10000,
    #max_test_samples=2000,
    max_train_samples=500,
    max_test_samples=200,
    checkpoint_path="artifacts/transformer_encoder_classifier.pt",
)


PREDICTION_CHECKPOINT_PATH = TRAINING_CONFIG.checkpoint_path
PREDICTION_DEVICE = TRAINING_CONFIG.device


def main() -> None:
    print_project_overview()

    if RUN_MODE == "smoke_test":
        logit_shape, num_attention_layers = run_smoke_test()
        print_smoke_test_result(logit_shape, num_attention_layers)
        return

    if RUN_MODE == "train":
        run_training(TRAINING_CONFIG)
        return

    if RUN_MODE == "predict":
        result = predict_text(
            text=PREDICTION_TEXT,
            checkpoint_path=PREDICTION_CHECKPOINT_PATH,
            device=PREDICTION_DEVICE,
        )
        print_prediction_result(result)
        return

    raise ValueError("RUN_MODE must be one of: smoke_test, train, predict")


if __name__ == "__main__":
    main()
