import torch


def weighted_average(predictions, weights):

    assert predictions.shape == weights.shape

    weights = weights / (torch.sum(weights, dim=0) + 1e-9)
    average = torch.sum(weights * predictions, dim=0)

    return average
