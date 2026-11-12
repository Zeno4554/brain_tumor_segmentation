"""Optional candidate inference refinement; shipped inference is unchanged."""
import numpy as np
from scipy import ndimage as ndi


def remove_small_regions(prediction: np.ndarray, min_area: int) -> np.ndarray:
    """Remove 8-connected components smaller than min_area, independently per slice."""
    prediction = np.asarray(prediction, dtype=bool)
    if prediction.ndim not in (2, 3) or min_area < 0 or int(min_area) != min_area:
        raise ValueError('Expected HxW or slices x H x W and a nonnegative integer area')
    volume = prediction[None] if prediction.ndim == 2 else prediction
    result = np.zeros_like(volume)
    for i, image in enumerate(volume):
        labels, _ = ndi.label(image, np.ones((3, 3), bool))
        keep = np.bincount(labels.ravel()) >= min_area
        keep[0] = False
        result[i] = keep[labels]
    return result[0] if prediction.ndim == 2 else result


def seeded_region_growth(probabilities: np.ndarray, low: float, high: float = 0.70) -> np.ndarray:
    """Keep 8-connected low-threshold regions containing a high-threshold seed.

    Connectivity is confined to each slice. Consequently the set of positive
    slices is exactly the same as thresholding at high; pixel errors can change.
    This cannot recover a slice with no high-confidence seed.
    """
    if not 0 < low <= high < 1:
        raise ValueError('Require 0 < low <= high < 1')
    probabilities = np.asarray(probabilities)
    if probabilities.ndim not in (2, 3) or not np.isfinite(probabilities).all():
        raise ValueError('Expected finite HxW or slices x H x W probabilities')
    volume = probabilities[None] if probabilities.ndim == 2 else probabilities
    result = np.zeros(volume.shape, dtype=bool)
    for i, image in enumerate(volume):
        labels, count = ndi.label(image >= low, structure=np.ones((3, 3), bool))
        keep = np.zeros(count + 1, dtype=bool)
        keep[np.unique(labels[image >= high])] = True
        keep[0] = False
        result[i] = keep[labels]
    return result[0] if probabilities.ndim == 2 else result
