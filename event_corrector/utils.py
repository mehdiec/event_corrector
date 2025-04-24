import numpy as np
import skimage
from scipy.ndimage import convolve
from skimage.morphology import skeletonize
import networkx as nx
import matplotlib.pyplot as plt


def get_bounding_box_from_coords(coords, labels=None, shape= None):
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


    y_coords, x_coords = coords[0], coords[1]
    y_min, y_max = min(y_coords), max(y_coords)
    x_min, x_max = min(x_coords), max(x_coords)

    return int(y_min), int(y_max), int(x_min), int(x_max)

def get_bounding_box_from_labels(labels, touched_labels):
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

    outlines = np.zeros(masks.shape, bool)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines

    # Use skimage.segmentation.find_boundaries to get outlines
    outlines = skimage.segmentation.find_boundaries(masks, mode="outer", background=0)
    outlines = skimage.morphology.remove_small_holes(outlines, 10)
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


def plot_subgraph(
    graph, start_node, depth=2, time_depth=2, relation="both", plot_future_edges=False
):
    # Extract subgraph
    nodes = {start_node}
    edges = set()
    current_nodes = {start_node}
    current_time = start_node[0]

    # Get nodes and edges up to specified depth
    for _ in range(depth):
        next_nodes = set()

        for node in current_nodes:
            if relation in ["both", "successors"]:
                successors = list(graph.successors(node))
                # Only add successors within time_depth
                valid_successors = [
                    s for s in successors if 0 <= s[0] - current_time <= time_depth
                ]
                next_nodes.update(valid_successors)
                edges.update((node, succ) for succ in valid_successors)

            if relation in ["both", "predecessors"]:
                predecessors = list(graph.predecessors(node))
                # Only add predecessors within time_depth
                valid_predecessors = [
                    p for p in predecessors if 0 <= current_time - p[0] <= time_depth
                ]
                next_nodes.update(valid_predecessors)
                edges.update((pred, node) for pred in valid_predecessors)

        nodes.update(next_nodes)
        current_nodes = next_nodes

    # Create directed graph for visualization
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from(edges)

    # Group nodes by time
    time_groups = {}
    for node in nodes:
        t = node[0]
        if t not in time_groups:
            time_groups[t] = []
        time_groups[t].append(node)

    # Calculate positions using a layered approach
    pos = {}
    times = sorted(time_groups.keys())

    # First pass: assign initial positions based on connectivity
    for t_idx, t in enumerate(times):
        nodes_at_t = time_groups[t]
        y = -t_idx * 2.0  # Increased vertical spacing between layers

        # Create a weighted ordering based on both predecessor and successor connections
        node_weights = {}

        for node in nodes_at_t:
            weight = 0
            count = 0

            # Look at predecessors
            if t_idx > 0:
                for pred in subgraph.predecessors(node):
                    if pred in pos:
                        weight += pos[pred][0]  # Use x coordinate
                        count += 1

            # Look at successors (if already positioned)
            if t_idx < len(times) - 1:
                for succ in subgraph.successors(node):
                    if succ in pos:
                        weight += pos[succ][0]  # Use x coordinate
                        count += 1

            # If no connections, use node label as weight
            if count == 0:
                node_weights[node] = node[1]
            else:
                node_weights[node] = weight / count

        # Sort nodes by their weights
        nodes_at_t.sort(key=lambda n: node_weights[n])

        # Position nodes with increased spacing
        spacing = 20.0  # Increased horizontal spacing
        total_width = spacing * (len(nodes_at_t) - 1)

        for n_idx, node in enumerate(nodes_at_t):
            x = -total_width / 2 + n_idx * spacing
            # Handle case when there's only one node to avoid division by zero
            if total_width == 0:
                scaled_x = 0  # Center single node
            else:
                scaled_x = x / (total_width / 2)  # Normalize to [-1,1]

            # Normalize y coordinate based on total number of time layers
            max_time_depth = len(times) - 1
            scaled_y = y / (max_time_depth * 2.0) if max_time_depth > 0 else 0

            pos[node] = (scaled_x, scaled_y)

    fig, ax = plt.subplots(figsize=(12, 16))

    for edge in edges:
        start, end = edge
        time_diff = end[0] - start[0]

        # Skip future edges if not plotting them
        if time_diff > 1 and not plot_future_edges:
            continue

        # Calculate curvature based on horizontal distance
        x_diff = pos[end][0] - pos[start][0]
        base_rad = 0.2 if time_diff == 1 else 0.3
        rad = base_rad * (1 + abs(x_diff) / 16.0)  # Adjusted curvature scaling

        # Alternate curve direction based on position to reduce crossings
        rad = rad if x_diff >= 0 else -rad

        # Draw edge
        color = "gray" if time_diff == 1 else "red"
        style = "solid" if time_diff == 1 else "dashed"

        nx.draw_networkx_edges(
            subgraph,
            pos,
            edgelist=[edge],
            edge_color=color,
            style=style,
            arrows=True,
            arrowsize=50,
            ax=ax
        )

        # Add edge weight label
        edge_weight = graph.edges[edge].get("weight", "")
        if edge_weight:
            # Calculate edge midpoint for label placement
            x1, y1 = pos[start]
            x2, y2 = pos[end]
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2
            ax.text(
                x_mid,
                y_mid,
                f"{edge_weight:.2f}",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        node_color="lightblue",
        node_size=1000,
        alpha=0.9,
        edgecolors="black",
        margins=0.1,
        ax=ax
    )

    # Highlight start node
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=[start_node],
        node_color="yellow",
        node_size=1000,
        edgecolors="black",
        linewidths=2,
        margins=0.1,
        ax=ax
    )
    # Add labels
    labels = {node: f"{node[0]}, {node[1]}" for node in nodes}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)

    ax.set_title(f"Subgraph from node {start_node}", fontsize=12, pad=20)
    ax.axis("off")
    
    return fig, ax
