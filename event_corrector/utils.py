import numpy as np
import skimage
from scipy.ndimage import convolve, find_objects
from skimage.morphology import skeletonize

def get_bounding_box_from_coords(coords):
    """
    Calculates the bounding box coordinates for a given list of coordinates.

    Parameters
    ----------
    coords : List[Tuple[int, int]]
        List of (y, x) coordinate tuples.

    Returns
    -------
    Tuple[int, int, int, int]
        Tuple containing the (y_min, y_max, x_min, x_max) values of the bounding box.
    """
    y_coords, x_coords = zip(*coords)
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

    outlines = np.zeros(masks.shape, bool)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines

    # Use skimage.segmentation.find_boundaries to get outlines
    outlines = skimage.segmentation.find_boundaries(masks, mode="outer", background=0)
    return outlines


def create_outline_from_mask(mask: np.ndarray) -> np.ndarray:
    """Create a skeletonized outline from a segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        Input segmentation mask

    Returns
    -------
    np.ndarray
        Binary outline image
    """
    # Getting rid of cellpose effect
    first_outline = masks_to_outlines(masks=mask)
    new_test = skimage.morphology.dilation(first_outline)
    new_test = skimage.morphology.dilation(new_test)
    skeleton_img = skimage.morphology.skeletonize(new_test)

    # Removing last impurities
    skeleton_img = skimage.morphology.remove_small_holes(skeleton_img, 20)
    skeleton_img = skimage.morphology.skeletonize(skeleton_img)
    skeleton_img = prune_skeleton(skeleton_img)

    # Label the connected components (8-connectivity)
    labelled_outline = skimage.measure.label(
        skeleton_img,
        connectivity=2,
    )

    # Create a mask that retains only the largest component (if needed)
    largest_component_outline = labelled_outline == 0

    return largest_component_outline


def process_seg_array(pipeline_seg: np.ndarray):
    out = ~np.array(pipeline_seg, dtype=bool)
    out = out.astype(np.uint8) * 255

    return out
