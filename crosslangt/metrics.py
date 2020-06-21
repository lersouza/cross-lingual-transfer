import torch

from sklearn.metrics import accuracy_score
from torch import Tensor


def compute_accuracy(logits: Tensor, y: Tensor) -> Tensor:
    """
    Calculates accuracy obtained from logits,
    based on actual `y` labels.
    """
    # Logits shape should be (B, C), where B is Batch Size
    # and C is the number of classes the model outputs.
    predicted = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(predicted.cpu(), y.cpu())

    return torch.tensor(accuracy)


