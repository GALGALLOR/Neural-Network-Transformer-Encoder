def print_project_overview() -> None:
    print("Transformer Encoder Classifier")
    print("Task: IMDB sentiment classification with manual self-attention and positional encoding")


def print_smoke_test_result(logit_shape, num_attention_layers: int) -> None:
    print(f"Smoke test logits shape: {tuple(logit_shape)}")
    print(f"Smoke test attention layers: {num_attention_layers}")


def print_prediction_result(result: dict[str, object]) -> None:
    print("Prediction Result")
    print(f"Sentiment: {result['predicted_sentiment']}")
    print(f"Positive probability: {result['probability_positive']:.4f}")
    print(f"Text: {result['text']}")
