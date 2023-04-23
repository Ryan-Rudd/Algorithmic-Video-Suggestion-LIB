import numpy as np


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    if np.count_nonzero(u) == 0 or np.count_nonzero(v) == 0:
        return 0.0
    else:
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def jaccard_similarity(u: set, v: set) -> float:
    if len(u) == 0 or len(v) == 0:
        return 0.0
    else:
        return len(u.intersection(v)) / len(u.union(v))

def pearson_correlation(u: np.ndarray, v: np.ndarray) -> float:
    u_mean = np.mean(u)
    v_mean = np.mean(v)
    if np.count_nonzero(u) == 0 or np.count_nonzero(v) == 0:
        return 0.0
    else:
        u_centered = u - u_mean
        v_centered = v - v_mean
        return np.dot(u_centered, v_centered) / (np.linalg.norm(u_centered) * np.linalg.norm(v_centered))

def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    return np.sqrt(np.sum(np.square(u - v)))

def manhattan_distance(u: np.ndarray, v: np.ndarray) -> float:
    return np.sum(np.abs(u - v))
