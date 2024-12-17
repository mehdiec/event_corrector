from collections import defaultdict
import networkx as nx
import numpy as np
import paramiko
import os
from pathlib import Path
from tqdm import tqdm


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
            stdout_data += stdout.channel.recv(1024).decode('utf-8')
        if stderr.channel.recv_stderr_ready():
            stderr_data += stderr.channel.recv_stderr(1024).decode('utf-8')
    
    # Get any remaining output
    stdout_data += stdout.read().decode('utf-8')
    stderr_data += stderr.read().decode('utf-8')

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
    threshold=0.5,
    edge_attr="weight",
):
    solution_graph = nx.DiGraph()
    # Group edges by their frame distance
    edges_by_distance = {}
    for edge in tqdm(
        candidate_graph.edges(data=True), desc="Grouping edges by distance"
    ):
        distance = edge[1][0] - edge[0][0]  # Frame distance between source and target
        if distance not in edges_by_distance:
            edges_by_distance[distance] = []
        edges_by_distance[distance].append(edge)

    # Sort each group by weight and combine into final list
    edges_by_distance = {
        k: sorted(v, key=lambda e: e[2]["weight"], reverse=True)
        for k, v in edges_by_distance.items()
    }

    for distance, edges in tqdm(
        edges_by_distance.items(), desc="Processing edges by distance"
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
                    if t_out - pred[0] == distance
                )
                if node_out in solution_graph
                else 0
            )
            number_outgoing_edges = (
                sum(
                    1
                    for succ in solution_graph.successors(node_in)
                    if succ[0] - t_in == distance
                )
                if node_in in solution_graph
                else 0
            )

            if distance == 1:
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
                if wt < threshold / 2:
                    break
                if node_out in solution_graph and number_incoming_edges > 0:
                    continue

                future_edge = True

            copy_edge(edge, candidate_graph, solution_graph, future_edge=future_edge)

    return solution_graph


def prediction_to_graph(predictions):
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
    return graph
    # Create new graph with (time,label) keys


def prediction_to_cell_lineage(predictions):
    graph = prediction_to_graph(predictions)
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


def nodes_to_event(graph):
    # Find nodes with 2 successors where either:
    # 1. One successor has no successors
    # 2. One successor has 2 successors
    # 3. Node has no successors and no predecessors
    # 4. Node has more than 2 successors
    # 5. Node has no direct successors but has successors in the next time frame
    nodes_to_plot_case_1 = []
    nodes_to_plot_case_2 = []
    nodes_to_plot_case_3 = []
    nodes_to_plot_case_4 = []
    nodes_to_plot_case_5 = []
    delamination = []
    new_cells = []
    divisions = []
    max_time = max(node[0] for node in graph.nodes())

    for node in tqdm(graph.nodes()):
        successors = get_direct_successors(graph, node)
        predecessors = get_direct_predecessors(graph, node)

        # Case 4: Node has more than 2 successors
        if len(successors) > 2:
            nodes_to_plot_case_4.append(node)

        # Case 5: Node has no direct successors but has successors in next frame
        elif len(successors) == 0 and len(list(graph.successors(node))) > 0:
            nodes_to_plot_case_5.append(node)

        # Case 3: Node has no connections
        elif len(successors) == 0 and len(predecessors) == 0:
            nodes_to_plot_case_3.append(node)

        # Delamination: Node has no successors
        elif len(successors) == 0 and node[0] < max_time:
            delamination.append(node)

        # New cells: Node has no predecessors
        elif len(predecessors) == 0 and node[0] > 0:
            new_cells.append(node)

        # Division cases
        elif len(successors) == 2:
            is_division = True
            for successor in successors:
                successor_successors = get_direct_successors(graph, successor)

                # Case 1: One successor has no successors
                if len(successor_successors) == 0:
                    nodes_to_plot_case_1.append(node)
                    is_division = False
                    break

                # Case 2: One successor has 2 successors
                elif len(successor_successors) == 2:
                    nodes_to_plot_case_2.append(node)
                    is_division = False
                    break

            if is_division:
                divisions.append(node)

    fraud_nodes = (
        nodes_to_plot_case_4
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

    label = skimage.io.imread("/home/mehdi/Documents/temp/fast1/mask/fast1_label.tif")

    with open("/home/mehdi/Documents/temp/fast1/pred.pkl", "rb") as f:
        predictions = pickle.load(f)
    cell_lineage = prediction_to_cell_lineage(predictions)
    events_labels = label_events(cell_lineage, label)
    print(events_labels)
