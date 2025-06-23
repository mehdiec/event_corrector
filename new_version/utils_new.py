import numpy as np
import skimage
from scipy.ndimage import convolve
from skimage.morphology import skeletonize
import networkx as nx
import matplotlib.pyplot as plt


def get_bounding_box_from_coord(coord, shape, bounding_box = 50):
    y_coord, x_coord = coord
    h,w = shape
    y_min, y_max = max(0,y_coord - bounding_box), min(h-1,y_coord+bounding_box)
    x_min, x_max = max(0,x_coord - bounding_box), min(w-1,x_coord+bounding_box)

    return int(y_min), int(y_max), int(x_min), int(x_max)

def get_bounding_box_from_coords(coords, shape, bounding_box = 50):
    y_coords, x_coords = coords[0], coords[1]
    h,w = shape
    y_min_coords, y_max_coords = min(y_coords), max(y_coords)
    x_min_coords, x_max_coords = min(x_coords), max(x_coords)

    y_min, y_max = max(0,y_min_coords - bounding_box), min(h-1,y_max_coords+bounding_box)
    x_min, x_max = max(0,x_min_coords - bounding_box), min(w-1,x_max_coords+bounding_box)

    return int(y_min), int(y_max), int(x_min), int(x_max)

def get_bounding_box_from_labels(labels, touched_labels):
    print("from labels")
    """
    Calculates the bounding box coordinates for a given list of labels.

    Returns
    -------
    Tuple[int, int, int, int]
        Tuple containing the (y_min, y_max, x_min, x_max) values of the bounding box.
    """
    

    if (labels is not None) and (touched_labels is not None):
        
        y_coords = []
        x_coords = []
        for label in touched_labels:
            label_mask = labels == label
            label_coords = np.where(label_mask)
            y_coords.extend(label_coords[0])
            x_coords.extend(label_coords[1])
                
        y_min, y_max = min(y_coords), max(y_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        
        return int(y_min), int(y_max), int(x_min), int(x_max)


def prune_skeleton(skel: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Iteratively prune endpoints from a skeleton until no endpoints remain or max iterations reached.

    Args:
        skel: Binary skeleton image
        max_iter: Maximum number of pruning iterations

    Returns:
        Pruned skeleton image
    """
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    for _ in range(max_iter):
        neighbor_count = convolve(
            skel.astype(np.uint8), kernel, mode="constant", cval=0
        )
        endpoints = neighbor_count == 11
        if not endpoints.any():
            break
        skel[endpoints] = 0
        skel = skeletonize(skel)

    return skel

def masks_to_outlines(masks: np.ndarray) -> np.ndarray:
    """Convert label masks to binary outlines.

    Args:
        masks: Label mask array of shape [Ly, Lx] or [Lz, Ly, Lx]

    Returns:
        Binary outline array of same shape as input

    Raises:
        ValueError: If input array dimension is not 2 or 3
    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError(
            f"masks_to_outlines takes 2D or 3D array, not {masks.ndim}D array"
        )
    print(masks.shape)
    outlines = np.zeros(masks.shape, bool)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines

    # Use skimage.segmentation.find_boundaries to get outlines
    outlines = skimage.segmentation.find_boundaries(masks, mode="outer", background=0)
    # outlines = skimage.morphology.remove_small_holes(outlines, 10)
    return outlines