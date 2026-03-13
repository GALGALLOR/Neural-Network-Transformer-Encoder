import torch

from src.controllers import predict_text
from src.views import print_prediction_result, print_project_overview


# Edit these values directly before running `python predict.py`.
PREDICTION_TEXT = "This movie kinda sucked but we watched all of it"
CHECKPOINT_PATH = "artifacts/transformer_encoder_classifier.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    print_project_overview()
    result = predict_text(
        text=PREDICTION_TEXT,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
    )
    print_prediction_result(result)


if __name__ == "__main__":
    main()
