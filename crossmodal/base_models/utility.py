import torch

def weighted_average(self, predictions, weights):
    assert predictions.shape == weights.shape

    weights = weights / (torch.sum(weights, dim=0) + 1e-9)
    weighted_average = torch.sum(weights * predictions, dim=0)

    return weighted_average
