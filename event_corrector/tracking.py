from collections import defaultdict
import networkx as nx
import numpy as np
import paramiko
import os
from pathlib import Path
from tqdm import tqdm


def assign_lineage_ids(graph, all_nodes):
    """Assigns unique IDs to cell lineages more efficiently"""

    # Pre-compute successors and predecessors for all nodes
    successor_map = {
        node: [s for s in graph.successors(node) if s[0] == node[0] + 1]
        for node in graph.nodes()
    }
    predecessor_map = {
        node: [p for p in graph.predecessors(node) if p[0] == node[0] - 1]
        for node in graph.nodes()
    }

    # Create frame buckets for better organization
    frames = {}
    for node in all_nodes:
        frame = node[0]
        if frame not in frames:
            frames[frame] = []
        frames[frame].append(node)

    next_id = 1
    id_map = {}  # Store node -> lineage_id mapping

    # Process frame by frame (more cache-friendly)
    for frame in tqdm(sorted(frames.keys()), desc="Processing frames"):
        for node in frames[frame]:
            if node in id_map:
                continue

            # If node has a predecessor, it should inherit that ID
            preds = predecessor_map.get(node, [])
            if preds:
                # Check if any predecessor has an ID
                pred_with_id = None
                for pred in preds:
                    if pred in id_map:
                        pred_with_id = pred
                        break

                if pred_with_id:
                    id_map[node] = id_map[pred_with_id]
                    continue

            # New lineage - assign new ID
            current_id = next_id
            next_id += 1

            # Follow lineage forward and assign same ID
            current = node
            while True:
                id_map[current] = current_id

                # Get successors
                succs = successor_map.get(current, [])
                if not succs:  # End of lineage
                    break

                # Handle multiple successors (division or complex case)
                if len(succs) > 1:
                    # Assign new IDs to all successors
                    for succ in succs:
                        if succ not in id_map:  # Only if not already assigned
                            next_id = assign_new_lineage(
                                succ, next_id, successor_map, id_map
                            )
                    break

                # Single successor case
                current = succs[0]
                if current in id_map:  # Stop if we hit an already processed node
                    break

    # Apply IDs to graph
    nx.set_node_attributes(graph, id_map, "absolute_number")
    return graph


def assign_new_lineage(start_node, next_id, successor_map, id_map):
    """Helper function to assign new IDs to a branch"""
    if start_node in id_map:  # Skip if already assigned
        return next_id

    current = start_node
    current_id = next_id

    while True:
        id_map[current] = current_id
        succs = successor_map.get(current, [])

        if not succs:  # End of lineage
            break

        if len(succs) > 1:  # Handle divisions
            next_id += 1
            for succ in succs:
                if succ not in id_map:
                    next_id = assign_new_lineage(succ, next_id, successor_map, id_map)
            break

        current = succs[0]
        if current in id_map:  # Stop if we hit an already processed node
            break

    return next_id + 1


def relabel_image(all_labels, solution_graph):
    # Using np.nonzero and ravel is faster than nested list comprehension
    all_nodes = []
    for frame_id in tqdm(range(all_labels.shape[0])):
        frame = all_labels[frame_id]
        labels = np.unique(frame[frame != 0])  # Get unique non-zero labels directly
        all_nodes.extend((frame_id, label) for label in labels)
    labels_new = np.zeros_like(all_labels)
    graph_abs_number = assign_lineage_ids(solution_graph, all_nodes)
    # Pre-compute node mappings for each frame
    frame_mappings = {}
    for node, data in graph_abs_number.nodes(data=True):
        if "absolute_number" in data:
            frame = node[0]
            label = node[1]
            if frame not in frame_mappings:
                frame_mappings[frame] = {}
            frame_mappings[frame][label] = data["absolute_number"]
        else:
            print(f"No absolute_number for node {node}")

    # Process each frame
    for frame_idx in tqdm(range(len(all_labels)), desc="Relabeling image"):
        # Create mapping array initialized with zeros (background)
        max_label = all_labels[frame_idx].max()
        label_map = np.zeros(max_label + 1, dtype=np.int32)

        # Fill mapping array from pre-computed dict
        if frame_idx in frame_mappings:
            for label, abs_num in frame_mappings[frame_idx].items():
                if label <= max_label:
                    label_map[label] = abs_num

        # Apply mapping in one vectorized operation
        labels_new[frame_idx] = label_map[all_labels[frame_idx]]
    return labels_new


def run_remote_tracking(
    host,
    user,
    password,
    remote_script_path,
    arg1,
    arg2,
):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password)

    command = f"/home/nexton/miniforge-pypy3/envs/trackastra/bin/python {remote_script_path} {arg1} {arg2}"
    stdin, stdout, stderr = client.exec_command(command)

    # Read and decode stdout/stderr while command executes
    stdout_data = ""
    stderr_data = ""
    while not stdout.channel.exit_status_ready():
        if stdout.channel.recv_ready():
            stdout_data += stdout.channel.recv(1024).decode("utf-8")
        if stderr.channel.recv_stderr_ready():
            stderr_data += stderr.channel.recv_stderr(1024).decode("utf-8")

    # Get any remaining output
    stdout_data += stdout.read().decode("utf-8")
    stderr_data += stderr.read().decode("utf-8")

    exit_status = stdout.channel.recv_exit_status()

    if exit_status != 0:
        client.close()
        raise RuntimeError(f"Error (exit code {exit_status}): {stderr_data}")

    print("Command output:")
    print(stdout_data)
    if stderr_data:
        print("Error output:")
        print(stderr_data)

    # Fetch the predictions file
    sftp = client.open_sftp()
    path_public = os.environ.get("path_public")
    remote_pred_path = Path(arg1).parent / "pred.pkl"
    local_pred_path = Path(arg1).parent / f"{Path(arg1).stem}_pred.pkl"

    try:
        sftp.get(str(remote_pred_path), str(local_pred_path))
        print(f"Downloaded predictions to {local_pred_path}")
    except FileNotFoundError as e:
        print(f"Error: Could not find remote predictions file {remote_pred_path}")
        raise e
    finally:
        sftp.close()
        client.close()

    # Load and return predictions locally
    import pickle

    with open(local_pred_path, "rb") as f:
        predictions = pickle.load(f)

    return predictions


def copy_edge(edge: tuple, source: nx.DiGraph, target: nx.DiGraph, future_edge=False):
    if edge[0] not in target.nodes:
        target.add_node(edge[0], **source.nodes[edge[0]])
    if edge[1] not in target.nodes:
        target.add_node(edge[1], **source.nodes[edge[1]])
    source.edges[(edge[0], edge[1])]["future_edge"] = future_edge
    target.add_edge(edge[0], edge[1], **source.edges[(edge[0], edge[1])])


def track_greedy(
    candidate_graph: nx.DiGraph,
    allow_divisions=True,
    threshold=0.6,
    edge_attr="weight",
):
    solution_graph = nx.DiGraph()
    # Group edges by their frame distance
    edges_by_dt = {}
    for edge in tqdm(
        candidate_graph.edges(data=True), desc="Grouping edges by distance"
    ):
        delta_t = edge[1][0] - edge[0][0]  # Frame distance between source and target
        if delta_t not in edges_by_dt:
            edges_by_dt[delta_t] = []
        edges_by_dt[delta_t].append(edge)

    # Sort each group by weight and combine into final list
    edges_by_dt = {
        k: sorted(v, key=lambda e: e[2]["weight"], reverse=True)
        for k, v in edges_by_dt.items()
    }

    for delta_t, edges in tqdm(
        edges_by_dt.items(), desc="Processing edges by distance"
    ):
        for edge in tqdm(edges, desc="Processing edges"):
            node_in, node_out, features = edge
            wt = features[edge_attr]
            t_out = node_out[0]
            t_in = node_in[0]
            number_incoming_edges = (
                sum(
                    1
                    for pred in solution_graph.predecessors(node_out)
                    if t_out - pred[0] == delta_t
                )
                if node_out in solution_graph
                else 0
            )
            number_outgoing_edges = (
                sum(
                    1
                    for succ in solution_graph.successors(node_in)
                    if succ[0] - t_in == delta_t
                )
                if node_in in solution_graph
                else 0
            )

            if delta_t == 1:
                if wt < threshold:
                    break
                if node_out in solution_graph.nodes and number_incoming_edges > 0:
                    # target node already has an incoming edge
                    continue

                if node_in in solution_graph and number_outgoing_edges >= (
                    2 if allow_divisions else 1
                ):
                    # parent node already has max number of outgoing edges
                    continue
                future_edge = False
            else:
                if wt < threshold:  # / 2:
                    break
                if node_out in solution_graph and number_incoming_edges > 0:
                    continue

                future_edge = True

            copy_edge(edge, candidate_graph, solution_graph, future_edge=future_edge)

    return solution_graph


def prediction_to_graph(predictions, labels):
    graph = nx.DiGraph()
    weights = {k: v for k, v in predictions["weights"]}

    # Add nodes with (time,label) keys
    for node in tqdm(predictions["nodes"], desc="Processing nodes"):
        time = node["time"]
        label = node["label"]
        coords = node["coords"]

        # Add node with (time,label) key and original attributes
        graph.add_node(
            (time, label),
            coords=coords,
            abs_number=(time, label),
            time=time,
            label=label,
        )

    # Add edges with weights
    for edge, weight in tqdm(weights.items(), desc="Adding edges"):
        source_node = predictions["nodes"][edge[0]]
        target_node = predictions["nodes"][edge[1]]

        source_key = (source_node["time"], source_node["label"])
        target_key = (target_node["time"], target_node["label"])

        graph.add_edge(
            source_key,
            target_key,
            weight=weight,
            future_edge=False if source_key[0] == target_key[0] - 1 else True,
        )
    for frame in tqdm(range(labels.shape[0]), desc="Adding nodes not in predictions"):
      
        for label in np.unique(labels[frame]):
            if label != 0 and (frame, label) not in graph.nodes:
                graph.add_node(
                    (frame, label),
                    coords=np.mean(np.where(labels[frame] == label), axis=1),
                    abs_number=(frame, label),
                    time=frame,
                    label=label,
                )

    return graph


def prediction_to_cell_lineage(predictions, labels):
    graph = prediction_to_graph(predictions, labels)
    return track_greedy(graph)


def get_direct_successors(graph, node):
    return [
        nodelette
        for nodelette in list(graph.successors(node))
        if nodelette[0] == node[0] + 1
    ]


def get_direct_predecessors(graph, node):
    return [
        nodelette
        for nodelette in list(graph.predecessors(node))
        if nodelette[0] == node[0] - 1
    ]


def count_predecessors(graph, node, limit=2):
    pred_count = 0
    current_nodes = [node]
    time = np.inf

    while current_nodes:
        n = current_nodes.pop()
        direct_preds = get_direct_predecessors(graph, n)

        for pred in direct_preds:
            if time > pred[0]:
                time = pred[0]
                if time == 0:
                    pred_count = 5
                    break
                pred_count += 1
                if pred_count > limit:
                    break
            current_nodes.append(pred)

        if pred_count > limit:
            break

    return pred_count


def nodes_to_event(graph):
    # Find nodes with 2 successors where either:
    # 1. One successor has no successors
    # 2. One successor has 2 successors
    # 3. Node has no successors and no predecessors
    # 4. Node has more than 2 successors
    # 5. Node has no direct successors but has successors in the next time frame
    # 6. Node has no successors and not enough predecessors
    # 7. Node has no successors but one of its predecessors has more than 
    #    1 succesor at an other timepoint
    nodes_to_plot_case_1 = []
    nodes_to_plot_case_2 = []
    nodes_to_plot_case_3 = []
    nodes_to_plot_case_4 = []
    nodes_to_plot_case_5 = []
    nodes_to_plot_case_6 = []
    nodes_to_plot_case_7 = []
    delamination = []
    new_cells = []
    divisions = []
    past_frauds = []
    number_of_predecesor_for_delamination = 4
    max_time = max(node[0] for node in graph.nodes())

    for node in tqdm(graph.nodes()):
        successors = get_direct_successors(graph, node)
        predecessors = get_direct_predecessors(graph, node)

        # Case 4: Node has more than 2 successors
        if len(successors) > 2:
            nodes_to_plot_case_4.append(node)
            if len(predecessors) > 0:
                past_frauds.append(predecessors[0])

        elif len(successors) == 2:
            is_division = True
            for successor in successors:
                successor_successors = get_direct_successors(graph, successor)

                # Case 1: One successor has no successors
                if len(successor_successors) == 0:
                    nodes_to_plot_case_1.append(node)
                    if len(predecessors) > 0:
                        past_frauds.append(predecessors[0])
                    is_division = False
                    break

                # Case 2: One successor has 2 successors
                elif len(successor_successors) == 2:
                    nodes_to_plot_case_2.append(node)
                    if len(predecessors) > 0:
                        past_frauds.append(predecessors[0])
                    is_division = False
                    break
                else:
                    for succ in successor_successors:
                        future_successors = get_direct_successors(graph, succ)
                        if len(future_successors) == 0:
                            nodes_to_plot_case_5.append(node)
                            is_division = False
                            break

            if is_division:
                divisions.append(node)

        # Case 5: Node has no direct successors but has successors in next frame
        elif len(successors) == 0 and len(list(graph.successors(node))) > 0:
            nodes_to_plot_case_5.append(node)
            if len(predecessors) > 0:
                past_frauds.append(predecessors[0])

        # Case 3: Node has no connections
        elif len(successors) == 0 and len(predecessors) == 0:
            nodes_to_plot_case_3.append(node)
            if len(predecessors) > 0:
                past_frauds.append(predecessors[0])

        # Delamination: Node has no successors
        elif len(successors) == 0 and node[0] < max_time:
            has_indirect_successors = len(list(graph.successors(node))) > 0
            number_of_pred = count_predecessors(
                graph, node, limit=number_of_predecesor_for_delamination
            )
            if number_of_pred < number_of_predecesor_for_delamination:
                nodes_to_plot_case_6.append(node)
                if predecessors:
                    past_frauds.append(predecessors[0])
            elif has_indirect_successors:
                delamination.append(node)
                successors_of_predecessors = graph.graph(predecessors[0])
                for successor in successors_of_predecessors:
                    if successor[0] == node[0]:
                        continue
                    if graph.edges[(predecessors[0], successor)]["weight"] > 0.6:
                        nodes_to_plot_case_7.append(node)
                        if predecessors:
                            past_frauds.append(predecessors[0])
                else:
                    if predecessors[0] not in nodes_to_plot_case_1:
                        delamination.append(node)
            else:
                if predecessors[0] not in (
                    nodes_to_plot_case_7
                    + nodes_to_plot_case_6
                    + nodes_to_plot_case_4
                    + nodes_to_plot_case_5
                    + nodes_to_plot_case_3
                    + nodes_to_plot_case_1
                    + nodes_to_plot_case_2
                ):
                    delamination.append(node)
        # New cells: Node has no predecessors
        elif len(predecessors) == 0 and node[0] > 0:
            new_cells.append(node)

        # Division cases

    fraud_nodes = (
        nodes_to_plot_case_7
        + nodes_to_plot_case_6
        + nodes_to_plot_case_4
        + nodes_to_plot_case_5
        + nodes_to_plot_case_3
        + nodes_to_plot_case_1
        + nodes_to_plot_case_2
    )

    return {
        "divisions": divisions,
        "delamination": delamination,
        "new_cells": new_cells,
        "frauds": fraud_nodes,
        "past_frauds": past_frauds,
    }


def label_events(cell_lineage, labels):
    events = nodes_to_event(cell_lineage)
    events_labels = defaultdict(lambda: np.zeros_like(labels))

    # Create boolean mask for all nodes at once
    for event_name, event_nodes in events.items():
        time_coords = np.array([node[0] for node in event_nodes])

        label_values = np.array([node[1] for node in event_nodes])
        # Use vectorized operations
        for t in tqdm(np.unique(time_coords), total=len(np.unique(time_coords))):
            t_mask = time_coords == t
            t_labels = label_values[t_mask]
            mask = np.isin(labels[t], t_labels)
            events_labels[event_name][t][mask] = 1

    return events_labels


if __name__ == "__main__":
    import pickle
    import skimage

    label = skimage.io.imread("/home/nexton/Documents/small_animal_ta/masks/mask_mbsRNAi_MOV2.tif")

    with open("/home/nexton/Documents/small_animal_ta/pred.pkl", "rb") as f:
        predictions = pickle.load(f)
    cell_lineage = prediction_to_cell_lineage(predictions, label)
    with open("/home/nexton/Documents/small_animal_ta/cell_lineage.pkl", "wb") as f:
        pickle.dump(cell_lineage, f)
    exit(0)
    events_labels = label_events(cell_lineage, label)
    print(events_labels)
