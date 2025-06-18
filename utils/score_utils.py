import numpy as np

# # calculate mean score for AVA dataset
# def mean_score(scores):
#     si = np.arange(1, 11, 1)
#     mean = np.sum(scores * si)
#     return mean

# # calculate standard deviation of scores for AVA dataset
# def std_score(scores):
#     si = np.arange(1, 11, 1)
#     mean = mean_score(scores)
#     std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
#     return std


def mean_score(scores: np.ndarray) -> float:
    """Compute mean aesthetic score as in the NIMA paper (∑i·pᵢ)."""
    values = np.arange(1, 11)
    return float(np.sum(values * scores))


def std_score(scores: np.ndarray) -> float:
    """Compute standard deviation of the predicted score distribution."""
    values = np.arange(1, 11)
    mean = mean_score(scores)
    return float(np.sqrt(np.sum(((values - mean) ** 2) * scores)))