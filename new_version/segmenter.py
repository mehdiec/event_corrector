import napari
from napari.utils.notifications import show_info, show_error
import skimage
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
)
from utils_new import (
    masks_to_outlines,
    get_bounding_box_from_coord,
    get_bounding_box_from_coords,
    get_bounding_box_from_labels
)
from skimage.measure import label as cc_label
import numpy as np
from skimage.draw import line


class HistoryManager:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def add_state(self, label_id, coords, before, after):
        # Save the state for undo functionality
        state = {
            "label_id": label_id,
            "coords": coords,  # (x_min, y_min, x_max, y_max)
            "before": before.copy(),
            "after": after.copy(),
        }
        self.undo_stack.append(state)
        self.redo_stack.clear()  # Clear redo stack on new action

    def undo(self):
        if self.undo_stack:
            state = self.undo_stack.pop()
            return state
        return None

    def redo(self):
        if self.redo_stack:
            state = self.redo_stack.pop()
            return state
        return None

    def save_for_redo(self, state):
        if state:
            self.redo_stack.append(state)

    def save_for_undo(self, state):
        if state:
            self.undo_stack.append(state)


class Segmenter:
    """
    Segmenter: An image annotation and segmentation tool.

    """

    def __init__(self, viewer: napari.Viewer, labels_layer, outlines_layer) -> None:

        self.viewer = viewer
        self.labels_layer = labels_layer
        self.outlines_layer = outlines_layer
        self.drawing = viewer.add_shapes(name="draw", blending="additive", shape_type="rectangle", edge_width=1)
        self.drawing_is_active = None
        self.shape = "Free Hand"
        self.history_manager = HistoryManager()
        self.connect_events()

    def connect_events(self):
        self.viewer.mouse_drag_callbacks.append(self.toggle_segmenting)
        self.viewer.mouse_drag_callbacks.append(self.remove_junction)
        self.viewer.mouse_drag_callbacks.append(self.fill_hole_on_click)
        self.viewer.mouse_drag_callbacks.append(self.remove_label)
        self.viewer.bind_key("C", self.clear_drawing, overwrite=True)
        self.viewer.bind_key("Control-Z", self.perform_undo, overwrite=True)
        self.viewer.bind_key("Control-Y", self.perform_redo, overwrite=True)
        self.viewer.bind_key("V", self.change_labels_visibility, overwrite=True)
        self.viewer.bind_key("O", self.change_outlines_visibility, overwrite=True)

    def get_labels_layer(self):
        return self.labels_layer
    
    def get_outlines_layer(self):
        return self.outlines_layer

    def set_labels_opacity(self, opacity):
        self.labels_layer.opacity = opacity
    
    def set_shape(self, shape):
        self.shape = shape
        
    def clean_holes(self, size_holes):
        for i in range(self.labels_layer.data.shape[0]):
            print(f"Cleaning frame {i}")
            frame = self.labels_layer.data[i]
            binary_objects = frame.astype(bool)
            binary_filled = skimage.morphology.remove_small_holes(binary_objects, size_holes)
            objects_filled = skimage.segmentation.watershed(
                    binary_filled, frame, mask=binary_filled
                    )
            self.labels_layer.data[i] = objects_filled
        self.labels_layer.refresh()

    def remove_junction(self, viewer, event):
        self.slider_pos = int(self.viewer.dims.point[0])
        frame = self.slider_pos
        print(f"Current frame: {frame}")
        if (
            QApplication.instance().keyboardModifiers() & Qt.ShiftModifier
            and event.button == 2
        ):
            coords = list(
                map(int, viewer.cursor.position[1:])
            )  # Convert to integer coordinates, skip z if necessary
            print(f"Click coordinates: {coords}")

            if len(coords) == 2:  # Adjust if your data is 2D
                    label_value = self.labels_layer.data[
                        self.slider_pos, coords[0], coords[1]
                    ]
                    print(f"Selected label value: {label_value}")

            x, y = coords
            radius = 3  
            num_points = 20  

            theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            circle_points = [(x + radius * np.cos(t), y + radius * np.sin(t)) for t in theta]
            coords_arr = np.array(circle_points, dtype=int)  
            ys = coords_arr[:, 0]
            xs = coords_arr[:, 1]
            shape = self.labels_layer.data[0].shape
            h, w = shape
            valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
            ys_valid = ys[valid]
            xs_valid = xs[valid]

            for i in range(self.slider_pos, self.slider_pos + 1):

                label_values = self.labels_layer.data[i][ys_valid, xs_valid]
                # Each unique labels
                uniques, counts = np.unique(label_values, return_counts=True)
                # We take only the top 2 in case the drawing overflow or is on a error solo pixel
                top2_idx   = np.argsort(counts)[::-1][:2]

                top2_labels = uniques[top2_idx]

                if top2_labels.size !=2:
                    continue
                
                # If the background is in the two labels, the bounding box is around the drawing
                if 0 in top2_labels:
                    top2_labels = top2_labels[top2_labels != 0]
                    y_min_draw, y_max_draw = ys_valid.min(), ys_valid.max()
                    x_min_draw, x_max_draw = xs_valid.min(), xs_valid.max()
                    y_min_labels, y_max_labels, x_min_labels, x_max_labels=get_bounding_box_from_labels(
                                self.labels_layer.data[i],
                                top2_labels
                            )
                    y_min, y_max = min(y_min_draw, y_min_labels), max(y_max_draw, y_max_labels)
                    x_min, x_max = min(x_min_draw, x_min_labels), max(x_max_draw, x_max_labels)
                    #We need 0 (background) to be at the end of the list
                    top2_labels = [top2_labels[0], 0]

                #If there is only labels and no background, the bounding box is around the labels
                else:
                    y_min, y_max, x_min, x_max = get_bounding_box_from_labels(self.labels_layer.data[i], top2_labels)

                before = self.labels_layer.data[
                    i,
                    y_min : y_max + 1,
                    x_min : x_max + 1
                ].copy()

                padding = 2

                valid = (ys >= y_min) & (ys <= y_max+1) & (xs >= x_min) & (xs <= x_max+1)
                ys = ys[valid]
                xs = xs[valid]
                
                new_slice = self.labels_layer.data[i][y_min:y_max+1, x_min:x_max+1].copy()
                # Padding with background to avoid problems at the border of the bounding box
                new_slice_padded = np.pad(new_slice, pad_width=padding, mode="constant",constant_values = 0)

                # Every pixel with the first label get the relabeled with the second
                new_slice_padded[new_slice_padded == top2_labels[0]] = top2_labels[1]
                self.labels_layer.data[i][y_min:y_max+1, x_min:x_max+1]=new_slice_padded[padding:-padding, padding:-padding]
                self.labels_layer.refresh()

                # Patching for outline context
                outline_borders = 2
                y0 = max(0, y_min - outline_borders)
                y1 = min(self.labels_layer.data[i].shape[0], y_max + 1 + outline_borders)
                x0 = max(0, x_min - outline_borders)
                x1 = min(self.labels_layer.data[i].shape[1], x_max + 1 + outline_borders)

                # Outlines with more context
                new_outline_with_context = masks_to_outlines(self.labels_layer.data[i][y0:y1, x0:x1])
                self.outlines_layer.data[i][y0:y1, x0:x1] = new_outline_with_context
                self.outlines_layer.refresh()
                after = self.labels_layer.data[
                        i,
                        y_min : y_max + 1,
                        x_min : x_max + 1
                    ].copy()
                
                self.history_manager.add_state(
                    0, (self.slider_pos, x_min, y_min, x_max, y_max), before, after
                )
    
    def fill_hole_on_click(self, viewer, event):

        if (QApplication.instance().keyboardModifiers() & Qt.ShiftModifier
            and event.button == 1):

            for i in range(self.slider_pos, self.slider_pos+1):
                coords = list(
                    map(int, viewer.cursor.position[1:])
                )
                y, x = coords

                
                shape = self.labels_layer.data[i].shape
                h, w = shape
                if not (0 <= y < h and 0 <= x < w):
                    show_error(f"Click ({y}, {x}) is outside image bounds (0–{h-1}, 0–{w-1})")
                    return
                
                if self.labels_layer.data[i][y,x] != 0:
                    show_error(f"Click on label {self.labels_layer.data[i,y,x]}, not background")
                    # on a cliqué sur un pixel étiqueté, pas dans le fond
                    return
                
                y_min, y_max, x_min, x_max = get_bounding_box_from_coord(coords, shape, bounding_box=100)

                before = self.labels_layer.data[
                    i,
                    y_min : y_max + 1,
                    x_min : x_max + 1
                ].copy()
                
                new_slice = self.labels_layer.data[i][y_min:y_max+1, x_min:x_max+1].copy()

                background_mask = (new_slice == 0)
                comps_background= cc_label(background_mask, connectivity=1)

                comp_id = comps_background[y-y_min, x-x_min]
                                
                coords = np.argwhere(comps_background == comp_id)
                ys, xs = coords[:, 0], coords[:, 1]
                H, W = new_slice.shape
                
                if (ys.min() == 0 or ys.max() == H-1 or xs.min() == 0 or xs.max() == W-1):
                    show_error('No hole detected')
                    return

                new_id = int(self.labels_layer.data.max()) + 1
                new_slice[comps_background == comp_id] = new_id
                self.labels_layer.data[i][y_min:y_max+1, x_min:x_max+1]= new_slice
                self.labels_layer.refresh()

                after = self.labels_layer.data[
                    i,
                    y_min : y_max + 1,
                    x_min : x_max + 1
                ].copy()

                self.history_manager.add_state(
                        0, (i, x_min, y_min, x_max, y_max), before, after
                    )


    def toggle_segmenting(self, viewer, event):
        if event.button == 2:
            if not (QApplication.instance().keyboardModifiers() & Qt.ShiftModifier):
                if self.drawing_is_active:
                    self.drawing_is_active = False
                    if self.shape == "Remove Segmentation":
                        self.remove_segmentation()
                        self.clear_drawing()
                    else:
                        self.split_or_add_labels(bounding_box=50)
                        self.clear_drawing()

                # If we enter segmenting mode
                else:
                    self.drawing_is_active = True
                    self.segmenting_path = [
                        self.viewer.cursor.position[1:]
                    ] # We remove the z position
                    self.viewer.mouse_move_callbacks.append(self.update_drawing)
                    self.viewer.mouse_drag_callbacks.append(self.update_drawing)

    def update_drawing(self, viewer, event):
        if self.drawing_is_active:
            if self.shape == "Free Hand":
                if self.drawing_is_active:
                    # This draws the line
                    self.segmenting_path += [event.position[1:]]
                    self.drawing.data = [self.segmenting_path]
                if not self.drawing.shape_type == "path":
                    self.drawing.shape_type = "path"
                # Set edge width to 2 pixels for thicker path
                self.drawing.edge_width = 1
            elif self.shape == "Line":
                # This draws the line
                if event.button == 1:
                    self.segmenting_path += [event.position[1:]]
                    self.drawing.data = [self.segmenting_path]
                    # if not self.drawing.shape_type == "line":
                    if not self.drawing.shape_type == "path":
                        self.drawing.shape_type = "path"
                    self.drawing.edge_width = 1
                    self.drawing.refresh()
            elif self.shape == "Remove Segmentation":
                y0, x0 = self.segmenting_path[0]
                y1, x1 = event.position[1:]

                self.drawing.data = [[(y0, x0), (y1, x1)]]

                if not self.drawing.shape_type == "rectangle":
                    self.drawing.shape_type = "rectangle"

                self.drawing.edge_width = 1

    def clear_drawing(self, viewer=None):
        if self.update_drawing in self.viewer.mouse_move_callbacks:
            self.viewer.mouse_move_callbacks.remove(self.update_drawing)
        if self.update_drawing in self.viewer.mouse_drag_callbacks:
                        self.viewer.mouse_drag_callbacks.remove(self.update_drawing)
        self.segmenting_path = []
        self.drawing.data = []
        self.drawing.refresh()
        self.drawing_is_active = False
        return

    def split_or_add_labels(self, bounding_box):
        if len(self.drawing.data[0]) > 0:
            if not (QApplication.instance().keyboardModifiers() & Qt.ShiftModifier):
                coords = np.array(self.drawing.data[0], dtype=int)

                ys_all, xs_all = [], []

                # Creates a line between the points of the drawing
                for (y0, x0), (y1, x1) in zip(coords[:-1], coords[1:]):
                    rr, cc = line(y0, x0, y1, x1)
                    ys_all.extend(rr)
                    xs_all.extend(cc)

                ys = np.array(ys_all)
                xs = np.array(xs_all)
                shape = self.labels_layer.data[0].shape
                h, w = shape
                valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
                ys_valid = ys[valid]
                xs_valid = xs[valid]
                #TODO
                for i in range(self.slider_pos, self.slider_pos + 1):

                    label_values = self.labels_layer.data[i][ys_valid, xs_valid]
                  
                    label_values_unique, counts = np.unique(label_values, return_counts = True)

                    # Bounding box is defined by the drawing and the labels touched (except background)
                    
                    # If the background is not the principal label
                    if np.argsort(counts)[::-1][0]!=0:
                        y_min_draw, y_max_draw = ys_valid.min(), ys_valid.max()
                        x_min_draw, x_max_draw = xs_valid.min(), xs_valid.max()
                        y_min_labels, y_max_labels, x_min_labels, x_max_labels=get_bounding_box_from_labels(
                                self.labels_layer.data[i],
                                label_values_unique[label_values_unique!=0]
                            )
                        y_min, y_max = min(y_min_draw, y_min_labels), max(y_max_draw, y_max_labels)
                        x_min, x_max = min(x_min_draw, x_min_labels), max(x_max_draw, x_max_labels)
                    # If the background is the principal label
                    else:
                        y_min, y_max, x_min, x_max = get_bounding_box_from_coords([ys_valid,xs_valid], shape)
                    
                    
                
                    before = self.labels_layer.data[
                        i,
                        y_min : y_max + 1,
                        x_min : x_max + 1
                    ].copy()

                    if (x_min==0) or (y_min==0) or (y_max==h-1) or (x_max==w-1):
                        padding = 0
                    else:
                        padding = 2

                    valid = (ys >= y_min) & (ys <= y_max) & (xs >= x_min) & (xs <= x_max)
                    ys = ys[valid]
                    xs = xs[valid]
                    
                    offset_y = y_min 
                    offset_x = x_min
                    local_ys = ys - offset_y + padding
                    local_xs = xs - offset_x + padding

                    new_slice = self.labels_layer.data[i][y_min:y_max+1, x_min:x_max+1].copy()
                    # Padding with background/labels to avoid problems at the border of the bounding box
                    new_slice_padded = np.pad(new_slice, pad_width=padding, mode="constant", constant_values=0)
                    
                    # Get a label id unused
                    new_id = self.labels_layer.data.max() + 1

                    # For each label touched, if it is entirely cut by the drawing -> 1 new label on only one side of the drawing, the other is untouched
                    for label in label_values_unique:
                        region  = (new_slice_padded == label)
                        barrier = np.zeros_like(region)
                        barrier[local_ys, local_xs] = True
                        mask_cut = region & (~barrier)
                        comps = cc_label(mask_cut, connectivity=1)

                        if comps.max() < 2:
                            print(f"No cut detected in label {label}")
                        # Relabelling new cell, only if size > 1 pix (to deal with solo error pixel)
                        else:
                            comps2 = [lab for lab in np.unique(comps) if lab >= 2]
                            for c in comps2:
                                comp_mask = (comps == c)
                                comp_size = comp_mask.sum()
                                if comp_size > 1:
                                    barrier_inside = barrier & region
                                    combined_mask = comp_mask | barrier_inside
                                    new_slice_padded[combined_mask] = new_id
                                    new_id += 1
                            else:
                                print(f"No cut detected in label {label}")

                    if padding !=0:
                        self.labels_layer.data[i][y_min:y_max+1, x_min:x_max+1]= new_slice_padded[padding:-padding, padding: -padding]
                    else:
                        self.labels_layer.data[i][y_min:y_max+1, x_min:x_max+1]= new_slice_padded

                    # Patching for outline context
                    outline_borders = 2
                    y0 = max(0, y_min - outline_borders)
                    y1 = min(self.labels_layer.data[i].shape[0], y_max + 1 + outline_borders)
                    x0 = max(0, x_min - outline_borders)
                    x1 = min(self.labels_layer.data[i].shape[1], x_max + 1 + outline_borders)

                    # Outlines with more context
                    new_outline_with_context = masks_to_outlines(self.labels_layer.data[i][y0:y1, x0:x1])
                    self.outlines_layer.data[i][y0:y1, x0:x1] = new_outline_with_context
                    

                    after = self.labels_layer.data[
                            i,
                            y_min : y_max + 1,
                            x_min : x_max + 1
                        ].copy()
                    
                    self.history_manager.add_state(
                        0, (i, x_min, y_min, x_max, y_max), before, after
                    )

        self.outlines_layer.refresh()
        self.labels_layer.refresh()

    def remove_label(self, viewer, event):
        """
        Handle mouse press events to delete labels with Ctrl + left click.
        """
        # Check if Ctrl is pressed and the left mouse button was clicked
        self.slider_pos = int(self.viewer.dims.point[0])
        frame = self.slider_pos
        print(f"Current frame: {frame}")

        if (
            QApplication.instance().keyboardModifiers() & Qt.ControlModifier
            and event.button == 1
        ):
            coords = list(
                map(int, viewer.cursor.position[1:])
            )  # Convert to integer coordinates, skip z if necessary
            print(f"Click coordinates: {coords}")

            if len(coords) == 2:  # Adjust if your data is 2D
                label_value = self.labels_layer.data[
                    self.slider_pos, coords[0], coords[1]
                ]
                print(f"Selected label value: {label_value}")

                if label_value != 0:
                    # Get mask of pixels with this label value
                    label_mask = self.labels_layer.data[self.slider_pos] == label_value
                    coords = np.where(label_mask)
                    y_coords, x_coords = coords[0], coords[1]
                    y_min, y_max = min(y_coords), max(y_coords)
                    x_min, x_max = min(x_coords), max(x_coords)

                    # Store state before modification
                    before = self.labels_layer.data[
                        self.slider_pos, y_min : y_max + 1, x_min : x_max + 1
                    ].copy()

                    # Update the data array directly
                    current_slice = self.labels_layer.data[self.slider_pos]
                    current_slice[label_mask] = 0
                    self.labels_layer.data[self.slider_pos] = current_slice

                    # Store state after modification
                    after = self.labels_layer.data[
                        self.slider_pos, y_min : y_max + 1, x_min : x_max + 1
                    ].copy()
                    
                    print("Label removed successfully")
                    show_info("Label removed successfully")

                    self.history_manager.add_state(
                        label_value, (frame, x_min, y_min, x_max, y_max), before, after
                    )

                    self.labels_layer.refresh()
                    # Patching for outline context
                    outline_borders = 2
                    y0 = max(0, y_min - outline_borders)
                    y1 = min(self.labels_layer.data[self.slider_pos].shape[0], y_max + 1 + outline_borders)
                    x0 = max(0, x_min - outline_borders)
                    x1 = min(self.labels_layer.data[self.slider_pos].shape[1], x_max + 1 + outline_borders)

                    # Outlines with more context
                    new_outline_with_context = masks_to_outlines(self.labels_layer.data[self.slider_pos][y0:y1, x0:x1])
                    self.outlines_layer.data[self.slider_pos][y0:y1, x0:x1] = new_outline_with_context

                    self.outlines_layer.refresh()
                    print("History updated and display refreshed")

    def remove_segmentation(self):
        if len(self.drawing.data[0]) > 0:
            ys, xs = self.drawing.data[0][:,0], self.drawing.data[0][:,1]
            ys = np.array(ys, dtype=int)
            xs = np.array(xs, dtype=int)
            shape = self.labels_layer.data[self.slider_pos].shape
            h, w = shape
            y_min, y_max = max(min(ys),0), min(max(ys), h)
            x_min, x_max = max(min(xs),0), min(max(xs), w)
            label_values = self.labels_layer.data[self.slider_pos][y_min:y_max, x_min:x_max]
            # Each unique labels
            uniques = np.unique(label_values)
            y_min, y_max = max(min(ys)-50,0), min(max(ys)+50,h-1)
            x_min, x_max = max(min(xs)-50,0), min(max(xs)+50,w-1)
            
            # Store state before modification
            before = self.labels_layer.data[
                self.slider_pos, y_min : y_max + 1, x_min : x_max + 1
            ].copy()

            labels = self.labels_layer.data[self.slider_pos, y_min : y_max + 1, x_min : x_max + 1]
            mask = np.isin(labels, uniques)
            labels[mask]=0
            self.labels_layer.data[self.slider_pos, y_min : y_max + 1, x_min : x_max + 1] = labels

            # Patching for outline context
            outline_borders = 2
            y0 = max(0, y_min - outline_borders)
            y1 = min(self.labels_layer.data[self.slider_pos].shape[0], y_max + 1 + outline_borders)
            x0 = max(0, x_min - outline_borders)
            x1 = min(self.labels_layer.data[self.slider_pos].shape[1], x_max + 1 + outline_borders)

            # Outlines with more context
            new_outline_with_context = masks_to_outlines(self.labels_layer.data[self.slider_pos][y0:y1, x0:x1])
            self.outlines_layer.data[self.slider_pos][y0:y1, x0:x1] = new_outline_with_context
            

            after = self.labels_layer.data[
                    self.slider_pos,
                    y_min : y_max + 1,
                    x_min : x_max + 1
                ].copy()
            
            self.history_manager.add_state(
                0, (self.slider_pos, x_min, y_min, x_max, y_max), before, after
            )

            self.outlines_layer.refresh()
            self.labels_layer.refresh()

    def change_outlines_visibility(self, viewer):
        if self.outlines_layer is not None:
            self.outlines_layer.visible = not self.outlines_layer.visible  
    
    def change_labels_visibility(self, viewer):
        if self.labels_layer is not None:
            self.labels_layer.visible = not self.labels_layer.visible  

    def perform_undo(self, viewer):
        state = self.history_manager.undo()
        if state:
            self.apply_state(state, undo=True)
            return
        print("Nothing to undo")
        show_info("Nothing to undo")

    def perform_redo(self, viewer):
        state = self.history_manager.redo()
        if state:
            self.apply_state(state, undo=False)
            return
        print("Nothing to redo")
        show_info("Nothing to redo")

    def apply_state(self, state, undo=False):
        # Apply changes from the state
        label_id, (frame, x_min, y_min, x_max, y_max), before, after = (
            state["label_id"],
            state["coords"],
            state["before"],
            state["after"],
        )
        if undo:
            self.labels_layer.data[frame, y_min : y_max + 1, x_min : x_max + 1] = before

            # Outlines with more context
            outline_borders = 2
            y0 = max(0, y_min - outline_borders)
            y1 = min(self.labels_layer.data[frame].shape[0], y_max + 1 + outline_borders)
            x0 = max(0, x_min - outline_borders)
            x1 = min(self.labels_layer.data[frame].shape[1], x_max + 1 + outline_borders)

            new_outline_with_context = masks_to_outlines(self.labels_layer.data[frame][y0:y1, x0:x1])
            self.outlines_layer.data[frame,y0:y1, x0:x1] = new_outline_with_context
            self.outlines_layer.data[frame, y_min : y_max + 1, x_min : x_max + 1] = (
                masks_to_outlines(before)
            )
            # Save the inverse operation for possible redo
            self.history_manager.save_for_redo(
                {
                    "label_id": label_id,
                    "coords": (frame, x_min, y_min, x_max, y_max),
                    "before": after,
                    "after": before,
                }
            )
        else:
            self.labels_layer.data[frame, y_min : y_max + 1, x_min : x_max + 1] = before

            # Outlines with more context
            outline_borders = 2
            y0 = max(0, y_min - outline_borders)
            y1 = min(self.labels_layer.data[frame].shape[0], y_max + 1 + outline_borders)
            x0 = max(0, x_min - outline_borders)
            x1 = min(self.labels_layer.data[frame].shape[1], x_max + 1 + outline_borders)

            new_outline_with_context = masks_to_outlines(self.labels_layer.data[frame][y0:y1, x0:x1])
            self.outlines_layer.data[frame,y0:y1, x0:x1] = new_outline_with_context
            self.outlines_layer.data[frame, y_min : y_max + 1, x_min : x_max + 1] = (
                masks_to_outlines(before)
            )
            self.history_manager.save_for_undo(
                {
                    "label_id": label_id,
                    "coords": (frame, x_min, y_min, x_max, y_max),
                    "before": after,
                    "after": before,
                }
            )

        self.labels_layer.refresh()
        self.outlines_layer.refresh()

    def __del__(self):
        print("Segmenter destroyed")