from pathlib import Path
from warnings import warn

import napari
import numpy as np
import skimage
import skimage.io
import zarr
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QCheckBox, QPushButton, QVBoxLayout, QWidget
from skimage.draw import polygon
from skimage.segmentation import find_boundaries


def is_shift_pressed():
    app = QApplication.instance()
    return app.keyboardModifiers() & Qt.ShiftModifier


def is_ctrl_click_pressed():
    app = QApplication.instance()
    return app.keyboardModifiers() & Qt.ControlModifier


def is_left_click_pressed():
    app = QApplication.instance()
    return app.mouseButtons() & Qt.LeftButton


def get_coordinates_for_label(labels_array, label_value):
    """
    Get separate x and y coordinates where the labels_array equals a specific label_value.

    :param labels_array: 2D numpy array of label data.
    :param label_value: The specific label value to find coordinates for.
    :return: Two arrays, coords_y and coords_x, containing the y and x coordinates respectively
             where the label equals label_value in the labels_array.
    """
    coords = np.where(labels_array == label_value)
    return coords


class SegmenterBindings:
    change_correcting_mode = "m"
    save_state = "s"
    cancel_drawing = "Escape"
    update_outline = "u"
    update_outline_alt = "Shift-u"
    is_ctrl_pressed = is_ctrl_click_pressed
    is_deletion_mode_activated = is_shift_pressed
    is_left_click_pressed = is_left_click_pressed
    undo = "z"
    free_hand = "f"


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
    if isinstance(coords, tuple):
        y_coords, x_coords = coords
    else:
        y_coords, x_coords = zip(*coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_min, x_max = min(x_coords), max(x_coords)

    return int(y_min), int(y_max), int(x_min), int(x_max)


def create_synthetic_data():
    # Create a small 3D image (t, h, w)
    image = np.random.rand(5, 100, 100) * 2
    image = image.astype(np.uint8)

    # Create corresponding labels
    mask = np.zeros((5, 100, 100), dtype=np.uint8)
    dividing = np.zeros_like(mask)
    dying = np.zeros_like(mask)

    # Draw some random circles as cells
    for i in range(5):
        for _ in range(10):  # 10 cells per frame
            r, c = np.random.randint(0, 100, 2)
            rad = np.random.randint(5, 15)
            rr, cc = skimage.draw.disk((r, c), rad, shape=mask[i].shape)
            mask[i][rr, cc] = i + 1  # Each cell has a unique label per frame

    # Randomly assign some cells as dividing or dying
    dividing[mask == np.random.randint(1, 11)] = 1
    dying[mask == np.random.randint(1, 11)] = 1

    return image, mask, dividing, dying


class CellClassificationWidget(QWidget):
    def __init__(self, viewer: napari.Viewer, animal_path, name):
        super().__init__()
        self.viewer = viewer

        self.animal_path = animal_path
        self.animal_zarr = zarr.open(animal_path)
        self.name = name

        self.history_manager = HistoryManager()
        # image_data = skimage.io.imread(path / f"{name}.tif")
        image_data = self.animal_zarr.IMAGE.D2.raw
        # mask = skimage.io.imread(path / f"masks/mask_{name}_raw.tif")
        # save_path_dying_correction = (
        #     self.animal_path / f"masks/apop_detection_{self.name}_correction.tif"
        # )
        # save_path_dividing_correction = (
        #     self.animal_path / f"masks/division_detection_{self.name}_correction.tif"
        # )

        # original_dying_path = self.animal_path / f"masks/apop_detection_{self.name}.tif"
        # original_dividing_path = (
        #     self.animal_path / f"masks/division_detection_{self.name}.tif"
        # )

        # if save_path_dying_correction.exists():
        #     dying = skimage.io.imread(save_path_dying_correction)
        # else:
        #     dying = skimage.io.imread(original_dying_path)

        # if save_path_dividing_correction.exists():
        #     dividing = skimage.io.imread(save_path_dividing_correction)
        # else:
        #     dividing = skimage.io.imread(original_dividing_path)

        labels = self.animal_zarr.IMAGE.D2.label[:]
        # mask = mask  # [:, y_start:y_end, x_start:x_end]

        # dividing = dividing == 1  # Assuming last but one channel are dividing cells
        # dying = dying == 1  # Assuming last channel are dying cells

        # Adding the image and labels layers to the viewer
        self.viewer.add_image(image_data, name="raw_image", channel_axis=1)
        self.labels_layer = self.viewer.add_labels(labels, name="labels")
        # self.layer_dividing = self.viewer.add_image(
        #     dividing.astype(np.int8),
        #     name="dividing_cells",
        #     blending="additive",
        #     # colormap="green",
        # )
        # self.layer_dying = self.viewer.add_image(
        #     dying.astype(np.int8),
        #     name="dying_cells",
        #     blending="additive",
        #     # colormap="cyan",
        # )
        # self.layer_dividing = self.viewer.add_labels(
        #     dividing.astype(np.int8), name="dividing_cells"
        # )
        # self.layer_dying = self.viewer.add_labels(
        #     dying.astype(np.int8), name="dying_cells"
        # )
        self.drawing = self.viewer.add_shapes(
            name="draw", blending="additive", shape_type="path"
        )

        # if os.path.exists(animal_path / f"masks/apop_pred_{name}.tif"):
        #     dying_proba = skimage.io.imread(animal_path / f"masks/apop_pred_{name}.tif")
        #     dividing_proba = skimage.io.imread(
        #         animal_path / f"masks/division_pred_{name}.tif"
        #     )
        #     self.viewer.add_image(
        #         dying_proba,
        #         name="dying probability",
        #         blending="additive",
        #         colormap="cyan",
        #     )
        #     self.viewer.add_image(
        #         dividing_proba,
        #         name="dividing probability",
        #         blending="additive",
        #         colormap="green",
        #     )

        self.masks = labels
        self.current_mode = "division"

        self.shape = "Free Hand"
        self.edition_mode = "automatic"

        self.segmenting_path = []
        self.slider_pos = int(self.viewer.dims.point[0])
        self.drawing_is_active = None

        # Set up layout and buttons
        self.display_cell_outlines(labels)
        self.setup()

    @staticmethod
    def extract_cell_outlines(img):
        # Find boundaries of segmented cells
        boundaries = find_boundaries(img, mode="outer")

        return boundaries

    def display_cell_outlines(self, labels):
        # Iterate over each time step
        outlines = []
        for i in range(labels.shape[0]):
            outlines.append(self.extract_cell_outlines(labels[i]))

        self.outline_layer = self.viewer.add_labels(
            np.array(outlines),
            name="outline ",
        )

    def update_cell_outlines(self):
        self.outline_layer.data[self.slider_pos] = self.extract_cell_outlines(
            self.labels_layer.data[self.slider_pos]
        )

    def setup(self):
        layout = QVBoxLayout()

        self.is_segmenting = False  # This flag will be toggled by the checkbox

        # Create a checkbox
        checkbox = QCheckBox("Enable Segmenting", self)
        checkbox.stateChanged.connect(self.toggle_segmenting)

        btn_add_division = QPushButton("Dividing Cell")
        btn_add_apoptosis = QPushButton("Apoptosis Cell")
        btn_save_annotations = QPushButton("Save Annotations")

        layout.addWidget(btn_add_division)
        layout.addWidget(btn_add_apoptosis)
        layout.addWidget(btn_save_annotations)
        layout.addWidget(checkbox)
        self.setLayout(layout)

        # Connect buttons to functions
        btn_add_division.clicked.connect(lambda: self.set_mode("division"))
        btn_add_apoptosis.clicked.connect(lambda: self.set_mode("apoptosis"))
        btn_save_annotations.clicked.connect(self.save_annotations)

        # Connect mouse click event to outline_cell function
        self.viewer.mouse_drag_callbacks.append(self.get_click_class)

        # Bind callback methods to mouse and key events
        self.viewer.mouse_move_callbacks.append(self.segmenting)
        self.viewer.mouse_drag_callbacks.append(self.on_segmenting)
        self.viewer.mouse_drag_callbacks.append(self.segmenting)
        self.viewer.mouse_drag_callbacks.append(self.remove_label)

        # Key bindings for various actions

        self.viewer.bind_key(
            SegmenterBindings.update_outline,
            self.add_label,
            overwrite=True,
        )
        self.viewer.bind_key(
            SegmenterBindings.update_outline_alt,
            self.add_label,
            overwrite=True,
        )

        self.viewer.bind_key("Control-Z", self.perform_undo)
        self.viewer.bind_key("Control-Y", self.perform_redo)

    def toggle_segmenting(self, state):
        # Update the is_segmenting flag based on the checkbox state
        self.is_segmenting = state == Qt.Checked
        print(f"Segmenting is {'enabled' if self.is_segmenting else 'disabled'}")

    def set_mode(self, mode):
        print(f"Mode set to {mode}")
        self.current_mode = mode

    def get_click_class(self, viewer, event):
        if self.is_segmenting:
            return
        dragged = False
        yield

        while event.type == "mouse_move":
            dragged = True
            yield

        if dragged:
            return

        self.class_assignation(viewer, event)

    def class_assignation(self, viewer, event):
        pass
        # Handling only left and right clicks
        # if event.button in {1, 2}:  # Left or right mouse button
        #     # try:
        #     # Getting the coordinates and the clicked label
        #     coords = np.round(viewer.cursor.position).astype(int)
        #     t, h, w = coords

        #     cell_label = self.masks[t, h, w]

        #     # Determine the layer to update based on the current mode
        #     if self.current_mode == "division" and event.button == 1:
        #         print(
        #             f"Adding dividing cell at position {coords} with label {cell_label}"
        #         )

        #         self.layer_dividing.data[t][self.masks[t] == cell_label] = 1
        #     elif self.current_mode == "apoptosis" and event.button == 1:
        #         print(f"Adding dying cell at position {coords} with label {cell_label}")
        #         self.layer_dying.data[t][self.masks[t] == cell_label] = 1
        #     elif (
        #         self.current_mode == "apoptosis" and event.button == 2
        #     ):  # Right click to remove from both
        #         print(f"Removing classification at position {coords}")
        #         self.layer_dying.data[t][self.masks[t] == cell_label] = 0

        #     elif (
        #         self.current_mode == "division" and event.button == 2
        #     ):  # Right click to remove from both
        #         print(f"Removing classification at position {coords}")

        #         self.layer_dividing.data[t][self.masks[t] == cell_label] = 0
        #     self.layer_dividing.refresh()
        #     self.layer_dying.refresh()  # Refresh both layers
        #     # except IndexError:
        #     #     print("Clicked outside the image boundaries.")

    def save_annotations(self):
        pass
        # save_path_dying = (
        #     self.animal_path / f"masks/apop_detection_{self.name}_correction.tif"
        # )
        # save_path_dividing = (
        #     self.animal_path / f"masks/division_detection_{self.name}_correction.tif"
        # )
        # save_path_mask = self.animal_path / f"masks/mask_{self.name}_correction.tif"
        # skimage.io.imsave(save_path_dividing, self.layer_dividing.data)
        # skimage.io.imsave(save_path_dying, self.layer_dying.data)
        # skimage.io.imsave(save_path_mask, self.labels_layer.data)
        # print("Annotations saved successfully.")

    def on_segmenting(self, viewer, event):
        """
        segmenting mode based on the middle mouse button.

        Parameters
        ----------
        viewer : object
            The viewer instance where images and shapes are displayed.
        event : object
            The event object containing information about the mouse event.

        Notes
        -----
        Assumes that `self.drawing_is_active` is initialized.
        """
        if not self.is_segmenting:
            return

        if event.button == 2:
            if self.drawing_is_active:
                self.drawing_is_active = False

                # Automatic mode : we add/remove the outlines instantaneously and clear the drawing
                if self.edition_mode == "automatic":
                    self.add_label()
                    self.clear_drawing()

            # If we enter segmenting mode
            else:
                self.drawing_is_active = True

                # If manual : we must redraw
                if self.edition_mode == "manual":
                    self.clear_drawing()

                self.segmenting_path = [
                    self.viewer.cursor.position[1:]
                ]  # We remove the z position

    def clear_drawing(self, viewer=None):
        """
        Reset the drawing and vertex state for the current segmentation operation.

        This method will clean up the current state of drawn paths

        Parameters
        ----------
        viewer : Optional[napari.viewer.Viewer]
            The Napari viewer instance. By default, it's set to None.
            This argument exists to match the expected signature
            for methods bound to viewer keypress events.

        Attributes Affected
        -------------------
        - drawing_is_active: Disables the freehand drawing mode.
        - segmenting_path: Clears the path of the segmenting drawing.
        - drawing.data: Resets the drawn shape data.
        """
        self.segmenting_path = []
        self.drawing.data = []
        self.drawing.refresh()
        self.drawing_is_active = False

    def segmenting(self, viewer, event):
        """
        Handles real-time drawing and interaction in segmenting mode.

        While in segmenting mode, this method updates the shape layer in the viewer
        to show the drawn path, checks for intersections with the outlines layer,
        and processes these intersections depending on the current edition mode.

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            The Napari viewer instance.
        event : Event
            An event triggered in the viewer.

        Attributes Affected
        -------------------
        - segmenting_path: Appends the current mouse position to the segmenting path.
        - drawing.data: Updates the drawing shape data.
        - current_tricellular_junctions: Appends detected intersection points.
        - outlines.data: Updates the drawn outlines data based on detected intersections.
        """
        if not self.is_segmenting:
            return
        # If the segmenting mode is activated, we update the shape layer at all times
        if self.drawing_is_active:
            if self.shape == "Free Hand":
                # if self.drawing_is_active:
                #     # This draws the line
                self.segmenting_path += [event.position[1:]]
                self.drawing.data = [self.segmenting_path]
                if not self.drawing.shape_type == "path":
                    self.drawing.shape_type = "path"
            elif self.shape == "Line":
                # This draws the line

                if event.button == 1:
                    self.segmenting_path += [event.position[1:]]
                    self.drawing.data = [self.segmenting_path]
                    # if not self.drawing.shape_type == "line":
                    if not self.drawing.shape_type == "path":
                        self.drawing.shape_type = "path"

            self.drawing.refresh()

    def add_label(self, viewer=None):
        """
        Perform image segmentation and modification operations in sequential mode.

        This function applies watershed segmentation and other morphological operations
        to a 3D stack of images, based on user-drawn lines and outlines.
        The function handles both deletion and addition modes to modify the outlines.

        Parameters
        ----------
        self : object
            The instance of the class that this method belongs to.
        viewer : object, optional
            An object representing the image viewer, default is None.
        bounding_box : int, optional
            The size of the bounding box around the drawn line for cropping, default is 200.
        padding : float, optional
            Additional padding added to the bounding box, default is 0.2.

        Notes
        -----
        The function relies on the following instance variables:
            - self.drawing.data: 2D numpy array representing the user-drawn lines
            - self.outlines.data: 3D numpy array representing the outlines
            - self.lower_change, self.upper_change: Integers representing the slice range for the 3D array

        The function modifies `self.outlines.data` in place.
        """

        if not self.drawing.data:
            warn(
                "On line left click for the first right click for the rest left click to end"
            )
            return

        if len(self.drawing.data[0]) > 0:
            # Get bounding box coordinates
            self.slider_pos = int(self.viewer.dims.point[0])

            # Process the slice currently in view
            current_slice = self.labels_layer.data[self.slider_pos]

            # Create a mask of the drawn area as labels
            coords_y, coords_x = zip(*self.drawing.data[0])
            y_min, y_max, x_min, x_max = get_bounding_box_from_coords(
                (self.drawing.data[0])
            )
            # coords_y = np.array(coords_y, dtype=np.int32)
            # coords_x = np.array(coords_x, dtype=np.int32)

            # Early closure detection: find a point close to the starting point to determine closure
            for i in range(1, len(coords_y)):
                distance_y = np.sqrt((coords_y[0] - coords_y[i]) ** 2)
                distance_x = np.sqrt((coords_x[0] - coords_x[i]) ** 2)
                distance = np.sqrt(
                    (coords_y[0] - coords_y[i]) ** 2 + (coords_x[0] - coords_x[i]) ** 2
                )
                if distance < 0.1 and distance_y > 0 and distance_x > 0:
                    coords_y = coords_y[: i + 1]
                    coords_x = coords_x[: i + 1]
                    break

            # Create a mask of the drawn area as labels
            rr, cc = polygon(coords_y, coords_x, current_slice.shape)

            # Ensure unique label
            new_label = current_slice.max() + 1

            # Update only those places in the mask where there are no pre-existing labels
            before = self.labels_layer.data[
                self.slider_pos, y_min : y_max + 1, x_min : x_max + 1
            ].copy()
            for r, c in zip(rr, cc):
                if (
                    current_slice[r, c] == 0
                ):  # Only update where there's no pre-existing label
                    current_slice[r, c] = new_label

            # Update the whole labels layer data for the slice
            self.labels_layer.data[self.slider_pos] = current_slice
            after = self.labels_layer.data[
                self.slider_pos, y_min : y_max + 1, x_min : x_max + 1
            ].copy()
            self.history_manager.add_state(
                new_label, (self.slider_pos, x_min, y_min, x_max, y_max), before, after
            )
            self.masks = self.labels_layer.data
            self.refresh_visualisation_layers()

    def remove_label(self, viewer, event):
        """
        Handle mouse press events to delete labels with Ctrl + left click.
        """
        # Check if Ctrl is pressed and the left mouse button was clicked
        self.slider_pos = int(self.viewer.dims.point[0])
        frame = self.slider_pos
        if (
            QApplication.instance().keyboardModifiers() & Qt.ControlModifier
            and event.button == 1
        ):
            coords = list(
                map(int, viewer.cursor.position[1:])
            )  # Convert to integer coordinates, skip z if necessary
            if len(coords) == 2:  # Adjust if your data is 2D
                label_value = self.labels_layer.data[
                    self.slider_pos, coords[0], coords[1]
                ]

                if label_value != 0:
                    coords = get_coordinates_for_label(
                        self.labels_layer.data[self.slider_pos], label_value
                    )
                    y_min, y_max, x_min, x_max = get_bounding_box_from_coords(coords)
                    # Set the label at this position to 0 (or another background value)
                    before = self.labels_layer.data[
                        self.slider_pos, y_min : y_max + 1, x_min : x_max + 1
                    ].copy()
                    self.labels_layer.data[self.slider_pos][
                        self.labels_layer.data[self.slider_pos] == label_value
                    ] = 0
                    after = self.labels_layer.data[
                        self.slider_pos, y_min : y_max + 1, x_min : x_max + 1
                    ].copy()
                    self.history_manager.add_state(
                        label_value, (frame, x_min, y_min, x_max, y_max), before, after
                    )
                    self.masks = self.labels_layer.data
                    self.refresh_visualisation_layers()

    def perform_undo(self, viewer):
        state = self.history_manager.undo()
        if state:
            self.apply_state(state, undo=True)

    def perform_redo(self, viewer):
        state = self.history_manager.redo()
        if state:
            self.apply_state(state, undo=False)
            return
        print("stop nothing to undo")

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
            self.history_manager.save_for_undo(
                {
                    "label_id": label_id,
                    "coords": (frame, x_min, y_min, x_max, y_max),
                    "before": after,
                    "after": before,
                }
            )

        self.refresh_visualisation_layers()
        self.masks = self.labels_layer.data

    def refresh_visualisation_layers(self):
        self.update_cell_outlines()
        self.labels_layer.refresh()
        self.outline_layer.refresh()


if __name__ == "__main__":
    # Create a Qt application
    app = QApplication([])

    # Create a napari viewer
    viewer = napari.Viewer()
    # animal_path = Path(
    #     "/Volumes/u934/equipe_bellaiche/m_ech-chouini/event_detection/validation/013"
    # )
    name = "013"
    animal_path = Path("/home/polina/Documents/data/210507_vi_pupa1")
    # Initialize our widget and pass the viewer
    cell_widget = CellClassificationWidget(viewer, animal_path, name)
    viewer.window.add_dock_widget(cell_widget)

    # Show the viewer and start the application
    napari.run()
