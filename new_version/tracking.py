import getpass
import os
import pickle
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import paramiko
import zarr
from skimage.morphology import binary_dilation
from skimage.segmentation import find_boundaries
from tqdm import tqdm


class CellTracker:
    def __init__(self, name_animal, raw_image):
        self.name_animal = name_animal
        self.image = raw_image
        self.event = None
        self.event_dictionnary = None
        self.relabeled_stuff = None

    def get_event(self):
        return self.event

    def get_dict(self):
        return self.event_dictionnary

    def get_relabeled(self):
        return self.relabeled_stuff

    def run(self, labels):
        if os.environ.get("USER") == "nexton":
            store = Path("/home/nexton/Documents/remote_tracking/nexton_user")
            os.makedirs(store, exist_ok=True)

            root = zarr.open_group(store=store, mode="r+")

            animal = root.require_group(self.name_animal)

            image_group = animal.require_group("IMAGE")
            d2_group = image_group.require_group("D2")

            d2_group.create_dataset(
                name="raw",
                data=self.image,
                chunks=(1,) + self.image.shape[1:],
                overwrite=True,
            )

            d2_group.create_dataset(
                name="label",
                data=labels,
                chunks=(1,) + labels.shape[1:],
                overwrite=True,
            )

            path_animal = store / self.name_animal

            command = (
                f"/home/nexton/miniforge-pypy3/envs/trackastra/bin/python "
                f"/home/nexton/Documents/trackastra-fusion/use_this_file_zarr.py "
                f"{path_animal}"
            )
            subprocess.run(command, shell=True, check=True)

            # Load predictions from the output file

            with open(path_animal / "pred.pkl", "rb") as f:
                predictions = pickle.load(f)
        else:
            predictions = run_remote_tracking(
                "10.50.11.184",
                "nexton",
                os.environ.get("nexton_password"),
                "/home/nexton/Documents/trackastra-fusion/use_this_file_zarr.py",
                self.image,
                labels,
                self.name_animal,
            )
        return predictions

    def run_cell_lineage(self, predictions, labels):
        cell_lineage = prediction_to_cell_lineage(predictions, labels[:])
        print("Cell lineage done")
        return cell_lineage


    def visualize_tracking_events(self, cell_lineage, labels, depth_frames):
        self.relabeled_stuff = relabel_image(labels, cell_lineage)
        print("Relabelling done")

        events_graph, self.event, self.event_dictionnary = label_events(
            cell_lineage, labels, depth_frames
        )

        return events_graph, self.relabeled_stuff, self.event_dictionnary


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
    host, user, password, remote_script_path, raw, labels, name_animal
):
    local_user = getpass.getuser()
    remote_base = Path(remote_script_path).parent.parent / "remote_tracking"
    remote_user_dir = remote_base / local_user

    # 1) SSH + SFTP
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password)
    sftp = client.open_sftp()

    try:
        sftp.mkdir(str(remote_user_dir))
    except IOError:
        pass

    from zarr.storage import FSStore

    path_store = "ssh://nexton@10.50.11.184" + str(remote_user_dir)
    print(path_store)
    store = FSStore(
        path_store,
        host="10.50.11.184",
        username="nexton",
        password=os.environ.get("nexton_password"),
    )
    root = zarr.open_group(store=store, mode="r+")
    animal = root.require_group(name_animal)

    image_group = animal.require_group("IMAGE")
    d2_group = image_group.require_group("D2")

    d2_group.create_dataset(
        name="raw", data=raw, chunks=(1,) + raw.shape[1:], overwrite=True
    )

    d2_group.create_dataset(
        name="label", data=labels, chunks=(1,) + labels.shape[1:], overwrite=True
    )

    path_animal = remote_user_dir / name_animal
    command = f"/home/nexton/miniforge-pypy3/envs/trackastra/bin/python {remote_script_path} {path_animal}"
    print(command)
    stdin, stdout, stderr = client.exec_command(command)

    # Read and decode stdout/stderr while command executes
    stdout_data = ""
    stderr_data = ""
    while not stdout.channel.exit_status_ready():
        if stdout.channel.recv_ready():
            output = stdout.channel.recv(1024).decode("utf-8", errors="replace")
            print(output, end="")
            stdout_data += output
            # stdout_data += stdout.channel.recv(1024).decode("utf-8")
        if stderr.channel.recv_stderr_ready():
            output = stderr.channel.recv_stderr(1024).decode("utf-8", errors="replace")
            # stderr_data += stderr.channel.recv_stderr(1024).decode("utf-8")
            print(output, end="")
            stderr_data += output

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
    remote_pred_path = path_animal / "pred.pkl"

    # Load and return predictions locally
    import pickle

    with sftp.open(str(remote_pred_path), "rb") as f:
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
    threshold_link=0.1,
    threshold_division=0.5,
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
            # x,y = graph.nodes[node_in]["coords"]
            # t = graph.nodes[node_in]["time"]
            # prob = div_raw[t,x,y]
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

            if number_incoming_edges > 0:
                    continue

            if delta_t == 1:
                
                if number_incoming_edges > 0:
                    continue
                elif number_outgoing_edges == 1:
                    if allow_divisions:
                        threshold = threshold_division
                        if wt < threshold:
                            continue
                    else:
                        continue
                else:
                    threshold = threshold_link
                    if wt < threshold:
                        break

                future_edge = False
            elif delta_t > 1:
                if wt < threshold_link:  # / 2:
                    break
                if number_incoming_edges > 0:
                    continue

                future_edge = True

            copy_edge(edge, candidate_graph, solution_graph, future_edge=future_edge)

    for node in candidate_graph.nodes:
        if node not in solution_graph:
            solution_graph.add_node(node)

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


@dataclass
class TrackingEvent:  # "division", "fake_fusion", "delamination", etc.
    nodes: List[Tuple[int, str]]  # liste de (frame, label) concernés
    predecessors: Optional[List[Tuple[int, str]]] = None
    successors: Optional[List[Tuple[int, str]]] = None  # pour fake_fusion, par ex.
    metadata: Dict[str, Any] = field(default_factory=dict)


def nodes_to_event_2(graph, depth_frames=10):
    events: Dict[str, List[TrackingEvent]] = defaultdict(list)
    number_of_predecesor_for_delamination = 4
    max_time = max(node[0] for node in graph.nodes())
    remaining = set(graph.nodes())
    division_attributes = {}
    error_attributes = {}
    new_attributes = {}
    delamination_attributes = {}

    for node in tqdm(graph.nodes(), desc="treating nodes"):
        if node not in remaining:
            continue
        node_time = node[0]
        successors = get_direct_successors(graph, node)
        predecessors = get_direct_predecessors(graph, node)
        indirect_predecessors = list(graph.predecessors(node))
        indirect_successors = list(graph.successors(node))
        has_indirect_predecessors = len(indirect_predecessors) > 0
        has_indirect_successors = len(indirect_successors) > 0

        if len(successors) == 2:
            valid_division = True
            successors_1_predecessors = list(graph.predecessors(successors[0]))
            successors_2_predecessors = list(graph.predecessors(successors[1]))

            one_cell_frames = [node]
            for depth in range(1, depth_frames + 1):
                if (node_time - depth) < 0:
                    break
                node_pred_1 = [
                    (t, label)
                    for t, label in successors_1_predecessors
                    if t == node_time - depth
                ]
                node_pred_2 = [
                    (t, label)
                    for t, label in successors_2_predecessors
                    if t == node_time - depth
                ]
                if (len(node_pred_1) == 1) and (len(node_pred_2) == 1):
                    if node_pred_1[0] != node_pred_2[0]:
                        if (
                            len(get_direct_successors(graph, node_pred_1[0])) == 0
                            or len(get_direct_successors(graph, node_pred_2[0])) == 0
                        ):
                            events["fake_fusion"].append(
                                TrackingEvent(
                                    nodes=one_cell_frames,
                                    predecessors=[node_pred_1[0], node_pred_2[0]],
                                )
                            )
                            for one_cell in one_cell_frames:
                                remaining.discard(one_cell)
                                error_attributes[one_cell] = 1
                            remaining.discard(node_pred_1[0])
                            remaining.discard(node_pred_2[0])
                            valid_division = False
                            break
                        else:
                            one_cell_frames.append(node_pred_1[0])
                    else:
                        one_cell_frames.append(node_pred_1[0])
                elif len(node_pred_1 + node_pred_2) == 1:
                    node_pred = node_pred_1 + node_pred_2
                    one_cell_frames.append(node_pred[0])

            # Each successor has a valid lineage of 1 successor in the next n frames
            if valid_division:
                node_successors = successors
                divisions_frames = [successors]
                for depth in range(depth_frames):
                    if node_time + 1 + depth >= max_time:
                        break
                    next_successors = []
                    for succ in node_successors:
                        successors_successor = get_direct_successors(graph, succ)
                        if len(successors_successor) == 1:
                            next_successors.append(successors_successor[0])
                    if len(next_successors) == 1:
                        one_cell_frames = [next_successors[0]]
                        solo_next_successors_fusion = next_successors[0]
                        for depth_fusion in range(depth_frames):
                            if solo_next_successors_fusion[0] >= max_time:
                                break
                            next_successors_fusion = get_direct_successors(
                                graph, solo_next_successors_fusion
                            )
                            if len(next_successors_fusion) == 2:
                                events["fake_fusion"].append(
                                    TrackingEvent(
                                        nodes=one_cell_frames,
                                        predecessors=divisions_frames[-1],
                                    )
                                )
                                for one_cell in one_cell_frames:
                                    remaining.discard(one_cell)
                                    error_attributes[one_cell] = 1
                                remaining.discard(next_successors_fusion[0])
                                remaining.discard(next_successors_fusion[1])
                                remaining.discard(divisions_frames[-1][0])
                                remaining.discard(divisions_frames[-1][1])
                                break
                            elif len(next_successors_fusion) == 0:
                                events["dying_successors"].append(
                                    TrackingEvent(nodes=[node])
                                )
                                error_attributes[node] = 3
                                valid_division = False
                                break
                            solo_next_successors_fusion = next_successors_fusion[0]
                            one_cell_frames.append(solo_next_successors_fusion)
                        else:
                            events["fake_division"].append(
                                TrackingEvent(
                                    nodes=divisions_frames,
                                    predecessors=[node],
                                    successors=one_cell_frames,
                                )
                            )
                            for one_cell in one_cell_frames:
                                remaining.discard(one_cell)
                            remaining.discard(node)
                            for two_cells in divisions_frames:
                                remaining.discard(two_cells[0])
                                remaining.discard(two_cells[1])
                                error_attributes[two_cells[0]]=2
                                error_attributes[two_cells[1]]=2
                            valid_division = False
                        break
                    elif len(next_successors) == 0:
                        valid_division = False
                        events["dying_successors"].append(TrackingEvent(nodes=[node]))
                        error_attributes[node] = 3
                        break
                    else:
                        divisions_frames.append(next_successors)
                        node_successors = next_successors
                        continue

            if valid_division:
                events["division"].append(TrackingEvent(nodes=[node]))
                remaining.discard(node)
                division_attributes[successors[0]] = True
                division_attributes[successors[1]] = True
                # divisions.append(node)

    for node in tqdm(remaining, desc="treating remaining nodes"):
        # if node not in remaining:
        #     continue
        node_time = node[0]
        successors = get_direct_successors(graph, node)
        predecessors = get_direct_predecessors(graph, node)
        indirect_predecessors = list(graph.predecessors(node))
        indirect_successors = list(graph.successors(node))
        has_indirect_predecessors = len(indirect_predecessors) > 0
        has_indirect_successors = len(indirect_successors) > 0

        if len(successors) == 0:
            # Case: Node has no connections
            if (not has_indirect_predecessors) and (not has_indirect_successors):
                events["no_lineage"].append(TrackingEvent(nodes=[node]))
                error_attributes[node]=4

            # Delamination: Node has no successors
            elif node_time != max_time:
                number_of_pred = count_predecessors(
                    graph, node, limit=number_of_predecesor_for_delamination
                )
                if number_of_pred < number_of_predecesor_for_delamination:
                    # dying_cells.append(node)
                    events["dying_cell"].append(TrackingEvent(nodes=[node]))
                    error_attributes[node]=5

                else:
                    if has_indirect_successors:
                        # delamination.append(node)
                        successors_of_predecessors = list(
                            graph.successors(predecessors[0])
                        )
                        for successor in successors_of_predecessors:
                            if successor[0] == node[0]:
                                continue
                            if (
                                graph.edges[(predecessors[0], successor)]["weight"]
                                > 0.6
                            ):
                                events["missed_successor"].append(
                                    TrackingEvent(nodes=[node], successors=[successor])
                                )
                                error_attributes[node]=6
                        else:
                            events["delamination"].append(TrackingEvent(nodes=[node]))
                            delamination_attributes[node]=True
                    else:
                        events["delamination"].append(TrackingEvent(nodes=[node]))
                        delamination_attributes[node]=True

        elif (node_time != 0) and (not has_indirect_predecessors):
            events["new_cell"].append(TrackingEvent(nodes=[node]))
            new_attributes[node]=True

    nx.set_node_attributes(graph, division_attributes, 'is_division')
    nx.set_node_attributes(graph, new_attributes, 'is_new')
    nx.set_node_attributes(graph, delamination_attributes, 'is_delamination')
    nx.set_node_attributes(graph, error_attributes, 'is_error')

    return graph, events


def label_events(cell_lineage, labels, depth_frames):
    graph, events = nodes_to_event_2(cell_lineage, depth_frames)
    # events_labels = {name: np.zeros_like(labels) for name in events}
    # events_labels = defaultdict(lambda: np.zeros_like(labels))

    # for event_name, ev_list in events.items():
    #     print(len(ev_list))
    #     times = []
    #     labs = []
    #     for ev in ev_list:
    #         seq = (
    #             ev.nodes
    #             if event_name != "fake_division"
    #             else [node for pair in ev.nodes for node in pair]
    #         )
    #         for t, lab in seq:
    #             times.append(t)
    #             labs.append(lab)

    #     if not times:
    #         continue

    #     times = np.array(times)
    #     labs = np.array(labs)

    #     mask_event = np.zeros_like(labels, dtype=np.uint8)

    #     for t in tqdm(np.unique(times), desc=event_name):
    #         mask_labels = np.isin(labels[t], labs[times == t])
    #         mask_event[t, mask_labels] = 1
    #         edges = find_boundaries(mask_event[t])
    #         structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    #         fine_edges = binary_dilation(edges, structure)
    #         events_labels[event_name][t, fine_edges] = 1

    #     # Libération de la mémoire
    #     del times, labs, mask_event

    mask_all = np.zeros_like(labels, dtype=np.uint8)
    codes = {name: idx for idx, name in enumerate(events, start=1)}

    for event_name, ev_list in events.items():
        code = codes[event_name]
        # … calcule mask_frame …
        
        times = []
        labs = []
        print(f"Number of {event_name}: {len(ev_list)}")

        for ev in ev_list:
            seq = (
                ev.nodes
                if event_name != "fake_division"
                else [node for pair in ev.nodes for node in pair]
            )
            for t, lab in seq:
                times.append(t)
                labs.append(lab)

        if not times:
            continue

        times = np.array(times)
        labs = np.array(labs)

        for t in tqdm(np.unique(times), desc=f"Labelling {event_name}"):
            mask_labels = np.isin(labels[t], labs[times == t])
            mask_frame = np.zeros_like(labels[t], dtype=np.uint8)
            mask_frame[mask_labels] = 1
            mask_all[t][mask_frame > 0] = code
            # events_labels[event_name][t] = mask_frame

        del times, labs

    return graph, events, mask_all

    # return events, events_labels
