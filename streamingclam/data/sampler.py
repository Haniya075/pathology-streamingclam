import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset

def weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    labels = np.array([int(label) for label in dataset.labels])

    # Check if labels are empty
    if labels.size == 0:
        raise ValueError("Dataset labels are empty.")
    
    # Calculate inverse class frequency and normalize
    _, class_counts = np.unique(labels, return_counts=True)
    
    if len(class_counts) == 1:
        raise ValueError("Dataset contains only one class. Cannot compute class weights.")
    
    inv_freq = len(labels) / class_counts
    norm_weights = inv_freq / np.max(inv_freq)
    
    # Create weight array and replace labels by their weights
    weights = np.array([norm_weights[label] for label in labels], dtype=np.float32)
    
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
