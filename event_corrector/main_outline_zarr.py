import pickle
from warnings import warn
from pathlib import Path
import napari
import numpy as np
import skimage.morphology
from tqdm import tqdm
import zarr
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
    QPushButton,
)
import os
import networkx as nx
from tracking import run_remote_tracking, prediction_to_cell_lineage, label_events
from utils import (
    create_outline_from_mask,
    process_seg_array,
    get_bounding_box_from_coords,
)
from binding import SegmenterBindings

os.environ["QT_QPA_PLATFORM"] = "xcb"
event_colors = {
    "divisions": "green",
    "delamination": "magenta",
    "new_cells": "cyan",
    "frauds": "red",
}


class SegmenterUI(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        """
        Initialize the Segmenter class with configurations for GUI and key bindings.

        Parameters
        ----------
        viewer : object
            The viewer instance where images and shapes are displayed.

        Notes
        -----
        This method assumes that various supporting methods and constants are defined
        elsewhere in the class. It configures sliders, text boxes, and key bindings for
        the segmentation interface.
        """

        super(QWidget, self).__init__()

        self.viewer = viewer

        self.slider_pos = int(self.viewer.dims.point[0])
        # Activate drawing and history layers

        self.viewer.dims.set_point(0, 0)

        # Initialize sliders and text boxes
        max_range = self.viewer.layers["raw"].data.shape[0]
        layout = QVBoxLayout(self)
        self.slider_layout = QHBoxLayout()
        text_layout = QHBoxLayout()

        self.draw_parameter_group_box = QGroupBox("Drawing Parameters")
        self.cell_tracking_group_box = QGroupBox("Run Cell Tracking")
        self.tracking_event_group_box = QGroupBox("Display Tracking event")
        self.export_results_group_box = QGroupBox("Export Results")

        self.cell_traking_box_layout = QVBoxLayout(self.cell_tracking_group_box)
        self.tracking_box_layout = QVBoxLayout(self.tracking_event_group_box)
        self.general_draw_parameter = QVBoxLayout(self.draw_parameter_group_box)
        self.export_results_layout = QVBoxLayout(self.export_results_group_box)

        # Initialize and configure the Export Segmentation button
        self.export_button = QPushButton("Export Segmentation")
        self.export_results_layout.addWidget(self.export_button)

        # Add the Export Results layout to the main layout
        layout.addWidget(self.export_results_group_box)

        self.lower_slider = self.init_slider(max_range - 1, "Lower:")
        self.upper_slider = self.init_slider(max_range + 1, "Upper:")
        self.general_draw_parameter.addLayout(self.slider_layout)
        self.lower_text = self.init_text_box(str(self.slider_pos))
        text_layout.addWidget(self.lower_text)
        self.upper_text = self.init_text_box(str(self.slider_pos + 1))
        text_layout.addWidget(self.upper_text)
        self.upper_change = self.slider_pos + 1
        self.lower_change = self.slider_pos
        self.lower_slider.valueChanged.connect(self.update_text_lower)
        self.upper_slider.valueChanged.connect(self.update_text_upper)
        self.lower_text.textChanged.connect(self.update_slider_lower)
        self.upper_text.textChanged.connect(self.update_slider_upper)
        self.general_draw_parameter.addLayout(text_layout)
        self.slider_enabled = False

        # Add a checkbox for marking segmentation as complete
        self.checkbox_multiple_modif = QCheckBox("multiple modification")

        self.general_draw_parameter.addWidget(self.checkbox_multiple_modif)

        # Further interface configurations
        self.possible_shapes = ["Free Hand", "Line"]
        self.shape = "Free Hand"
        self.combo_box_shape = QComboBox()
        for mode in self.possible_shapes:
            self.combo_box_shape.addItem(mode)

        self.general_draw_parameter.addWidget(self.combo_box_shape)

        # Add a checkbox for marking segmentation as complete
        self.checkbox = QCheckBox("Completed Segmentation")

        layout_complete_seg = QVBoxLayout()
        layout_complete_seg.addWidget(self.checkbox)
        self.general_draw_parameter.addLayout(layout_complete_seg)

        # self.general_draw_parameter.addLayout(self.general_draw_parameter)

        # Additional setup
        self._show_widget_cell_tracking()

        # self.cell_traking_box_layout.addLayout(self.cell_traking_box_layout)
        layout.addWidget(self.draw_parameter_group_box)
        layout.addWidget(self.cell_tracking_group_box)
        layout.addWidget(self.tracking_event_group_box)
        self.setLayout(layout)

        self.upper_bound = 0
        self.lower_bound = 0

    def init_slider(self, max_range, name):
        """
        Initialize a QSlider widget for the interface.

        Parameters
        ----------
        max_range : int
            The maximum range value for the slider.
        name : str
            The label to identify the slider.

        Returns
        -------
        slider : QSlider
            Initialized QSlider widget.
        """

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, max_range - 1)
        slider.setValue(self.slider_pos)
        self.slider_layout.addWidget(QLabel(name))
        self.slider_layout.addWidget(slider)
        return slider

    def init_text_box(self, initial_text):
        """
        Initialize a QLineEdit widget for the interface.

        Parameters
        ----------
        initial_text : str
            The initial text to populate the QLineEdit widget.

        Returns
        -------
        text_box : QLineEdit
            Initialized QLineEdit widget.
        """

        text_box = QLineEdit()
        text_box.setText(initial_text)
        return text_box

    def update_text_lower(self):
        """
        Update the lower bound text based on the lower slider's value.

        Notes
        -----
        This method assumes that `self.lower_slider` and `self.upper_change` are initialized.
        """

        new_value = min(self.lower_slider.value(), self.upper_change - 1)
        if self.lower_text.text() != str(new_value):
            self.lower_text.setText(str(new_value))
        self.lower_change = new_value

    def update_slider_lower(self):
        """
        Update the lower slider based on the lower text box's value.

        Notes
        -----
        This method assumes that `self.lower_text` and `self.upper_change` are initialized.
        """

        text_value = self.lower_text.text()
        if text_value.isdigit():
            new_value = min(int(text_value), self.upper_change - 1)
            if self.lower_slider.value() != new_value:
                self.lower_slider.setValue(new_value)
        self.lower_change = new_value

    def update_text_upper(self):
        """
        Update the upper bound text based on the upper slider's value.

        Notes
        -----
        This method assumes that `self.upper_slider` and `self.lower_change` are initialized.
        """

        new_value = max(self.upper_slider.value(), self.lower_change + 1)
        if self.upper_text.text() != str(new_value):
            self.upper_text.setText(str(new_value))
        self.upper_change = new_value

    def update_slider_upper(self):
        """
        Update the upper slider based on the upper text box's value.

        Notes
        -----
        This method assumes that `self.upper_text` and `self.lower_change` are initialized.
        """

        text_value = self.upper_text.text()
        if text_value.isdigit():
            new_value = max(int(text_value), self.lower_change + 1)
            if self.upper_slider.value() != new_value:
                self.upper_slider.setValue(new_value)
        self.upper_change = new_value

    def display_cell_tracking_event_widget(self):
        self.button_cell_event = QPushButton("Display tracking evnt")
        self.tracking_box_layout.addWidget(self.button_cell_event)

    def create_tracking_widget(self):
        self.button_cell_tracking = QPushButton("Run cell tracking")
        self.cell_traking_box_layout.addWidget(self.button_cell_tracking)

    def _show_widget_cell_tracking(self):
        self.tracking_widget = self.create_tracking_widget()
        self.cell_tracking_event_widget = self.display_cell_tracking_event_widget()


class Segmenter:
    """
    Segmenter: An image annotation and segmentation tool.

    """

    def __init__(self, viewer: napari.Viewer, animal_path) -> None:
        """
        Initialize the Segmenter class with configurations for GUI and key bindings.

        Parameters
        ----------
        viewer : object
            The viewer instance where images and shapes are displayed.
        animal_path : str
            The path to the animal data.

        Notes
        -----
        This method assumes that various supporting methods and constants are defined
        elsewhere in the class. It configures sliders, text boxes, and key bindings for
        the segmentation interface.
        """

        self.animal = zarr.open(animal_path)

        self.public_path = Path(os.environ.get("path_public"))
        self.image = self.animal.IMAGE.D2.raw

        self.labels = self.animal.IMAGE.D2.label
        self.cell_lineage = None

        if os.path.exists(Path(animal_path) / "pred.pkl"):
            with open(Path(animal_path) / "pred.pkl", "rb") as f:
                predictions = pickle.load(f)
            self.cell_lineage = prediction_to_cell_lineage(predictions)

        self.viewer: napari.Viewer = viewer

        #
        #     self.tissue_mask = skimage.io.imread(Path(animal_path) / "labels.tif")
        # except:
        # Create tissue mask by dilating labels to make cells touch, preserving time dimension

        self.tissue_mask = np.zeros_like(self.labels)
        for t in range(self.labels.shape[0]):
            self.tissue_mask[t] = skimage.morphology.dilation(
                self.labels[t] > 0, skimage.morphology.disk(1)
            )

        self.raw = self.viewer.add_image(
            np.array(self.image),
            name="raw",
            channel_axis=1,
        )

        skeleton = []
        for i in tqdm(range(self.labels.shape[0])):
            skeleton.append(create_outline_from_mask(self.labels[i]))
        self.skeleton = np.array(skeleton)
        self.outlines = self.viewer.add_labels(
            process_seg_array(self.skeleton), name="outlines", visible=True
        )

        self.cell_mask = self.viewer.add_labels(
            process_seg_array(self.tissue_mask), name="cell_mask", visible=True
        )

        self.labels_on_viewer = self.viewer.add_labels(
            (self.labels), name="labels", visible=False
        )

        self.viewer = viewer
        self.drawing_is_active = None
        self.history = []
        self.shape = "Free Hand"

        self.ui_widget = SegmenterUI(viewer)

        self.slider_pos = int(self.viewer.dims.point[0])
        self.drawing = self.viewer.add_shapes(
            name="draw", blending="additive", shape_type="path"
        )

        self.viewer.layers.selection.active = self.outlines

        self.edition_mode = "automatic"
        self.complete_segmentation = {}

        self.viewer.window.add_dock_widget(self.ui_widget)

        self.upper_bound = 0
        self.lower_bound = 0
        self._connect_widget()

    def _connect_widget(self):
        # Bind callback methods to mouse and key events
        self.viewer.mouse_move_callbacks.append(self.segmenting)
        self.viewer.mouse_drag_callbacks.append(self.toggle_segmenting)
        self.viewer.mouse_drag_callbacks.append(self.segmenting)

        # Key bindings for various actions
        self.viewer.bind_key(
            SegmenterBindings.cancel_drawing, self.clear_drawing, overwrite=True
        )
        self.viewer.bind_key(
            SegmenterBindings.save_state, self.on_export_current_labels, overwrite=True
        )
        self.viewer.bind_key(
            SegmenterBindings.update_outline,
            self.handle_sequential_mode,
            overwrite=True,
        )
        self.viewer.bind_key(
            SegmenterBindings.update_outline_alt,
            self.handle_sequential_mode,
            overwrite=True,
        )
        self.viewer.bind_key(
            SegmenterBindings.undo, self.undo_last_operation, overwrite=True
        )
        self.viewer.bind_key(
            SegmenterBindings.free_hand, self.switch_edition_mode, overwrite=True
        )

        # Apply key bindings to all layers
        for layer in self.viewer.layers:
            layer.bind_key(
                SegmenterBindings.cancel_drawing, self.clear_drawing, overwrite=True
            )
            layer.bind_key(
                SegmenterBindings.save_state,
                self.on_export_current_labels,
                overwrite=True,
            )
            layer.bind_key(
                SegmenterBindings.update_outline,
                self.handle_sequential_mode,
                overwrite=True,
            )
            layer.bind_key(
                SegmenterBindings.update_outline_alt,
                self.handle_sequential_mode,
                overwrite=True,
            )
            layer.bind_key(
                SegmenterBindings.undo, self.undo_last_operation, overwrite=True
            )
            layer.bind_key(
                SegmenterBindings.free_hand, self.switch_edition_mode, overwrite=True
            )
        self.ui_widget.checkbox_multiple_modif.stateChanged.connect(
            lambda state: self.enable_slide(state == Qt.Checked)
        )
        self.ui_widget.combo_box_shape.currentIndexChanged.connect(
            self.switch_edition_mode
        )
        # Add a checkbox for marking segmentation as complete
        self.ui_widget.checkbox.stateChanged.connect(
            lambda state: self.tag_frame(state == Qt.Checked)
        )
        self.viewer.dims.events.connect(self.update_slider)

        self.ui_widget.button_cell_tracking.clicked.connect(self.run_cell_tracking)
        self.ui_widget.button_cell_event.clicked.connect(self.visualize_tracking_events)

        self.ui_widget.export_button.clicked.connect(self.on_export_current_labels)

    def clean_state(self):
        """Removes layers from viewer"""

        layer_names = [layer.name for layer in self.viewer.layers]

        for layer in layer_names:
            self.viewer.layers.remove(layer)

        if self.ui_widget is not None:
            self.viewer.window.remove_dock_widget(self.ui_widget)
            self.ui_widget.deleteLater()
            self.viewer.dims.events.disconnect(self.update_slider)

            self.ui_widget = None
            # Disconnect all signal-slot connections of the widget
        # self.ui_widget = None

    def update_slider(self, event):
        """
        Update the slider position and related attributes based on viewer dimensions.

        Parameters
        ----------
        event : object
            The event object containing information about the event.

        Notes
        -----
        Assumes that `self.viewer.dims.point`, `self.lower_text`, and `self.upper_text` are initialized.
        """

        # Update slider and bounds based on viewer's current point
        self.slider_pos = int(self.viewer.dims.point[0])
        self.upper_bound = self.slider_pos + 1
        self.lower_bound = self.slider_pos
        self.ui_widget.lower_text.setText(str(self.lower_bound))
        self.ui_widget.upper_text.setText(str(self.upper_bound))

        # Update checkbox state based on new slider position
        self.ui_widget.checkbox.setChecked(
            self.complete_segmentation.get(str(self.slider_pos), False)
        )

    def tag_frame(self, checked):
        """
        Tag the current frame as either complete or incomplete.

        Parameters
        ----------
        checked : bool
            Whether the frame is complete (True) or not (False).

        Notes
        -----
        Updates the `complete_segmentation` attribute in the Zarr store.
        """

        self.complete_segmentation[str(self.slider_pos)] = checked

    def enable_slide(self, checked):
        """
        Tag the current frame as either complete or incomplete.

        Parameters
        ----------
        checked : bool
            Whether the frame is complete (True) or not (False).

        Notes
        -----
        Updates the `complete_segmentation` attribute in the Zarr store.
        """

        self.ui_widget.slider_enabled = checked

    def switch_edition_mode(self, viewer):
        """
        Switch between Free Hand and Line shapes for drawing.

        Parameters
        ----------
        viewer : object
            The viewer instance where images and shapes are displayed.

        Notes
        -----
        Assumes that `self.combo_box_shape` is initialized and contains the shape options.
        """

        self.ui_widget.combo_box_shape.blockSignals(True)
        self.shape = "Line" if self.shape == "Free Hand" else "Free Hand"
        self.ui_widget.combo_box_shape.setCurrentText(self.shape)
        self.ui_widget.combo_box_shape.blockSignals(False)

    def toggle_segmenting(self, viewer, event):
        """
        Toggle segmenting mode based on the middle mouse button.

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

        if event.button == 2:
            if self.drawing_is_active:
                self.drawing_is_active = False

                # Automatic mode : we add/remove the outlines instantaneously and clear the drawing
                if self.edition_mode == "automatic":
                    self.handle_sequential_mode()
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

    # To purge existing crossings and remove freehand
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
        self.outlines.refresh()
        self.drawing_is_active = False

    def undo_last_operation(self, viewer):
        if len(self.history) > 0:
            crop_change_outlines, crop_change_cell_mask, frames_bound, yy, xx = (
                self.history.pop()
            )

            lower_frame = frames_bound[0]
            upper_frame_frame = frames_bound[1]
            y_min_bound = yy[0]
            y_max_bound = yy[1]
            x_min_bound = xx[0]
            x_max_bound = xx[1]

            self.outlines.data[
                lower_frame:upper_frame_frame,
                y_min_bound + 5 : y_max_bound - 5 + 1,
                x_min_bound + 5 : x_max_bound - 5 + 1,
            ] = crop_change_outlines
            self.cell_mask.data[
                lower_frame:upper_frame_frame,
                y_min_bound + 5 : y_max_bound - 5 + 1,
                x_min_bound + 5 : x_max_bound - 5 + 1,
            ] = crop_change_cell_mask
            self.outlines.refresh()
            self.cell_mask.refresh()

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

    def handle_sequential_mode(self, viewer=None, bounding_box=200, padding=0.2):
        """
        Perform image segmentation and modification operations in sequential mode.

        This function applies watershed segmentation and other morphological operations
        to a 3D stack of images, based on user-drawn lines and outlines.
        The function handles both deletion and addition modes to modify the outlines.
        It also updates the cell mask by removing the drawn outlines from it.

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
            - self.cell_mask.data: 3D numpy array representing the tissue mask
            - self.lower_change, self.upper_change: Integers representing the slice range for the 3D array

        The function modifies `self.outlines.data` and `self.cell_mask.data` in place.
        """

        # Check if drawing data exists, if not warn the user

        if not self.ui_widget.slider_enabled:
            self.ui_widget.lower_change = self.slider_pos
            self.ui_widget.upper_change = self.slider_pos + 1

        if not self.drawing.data:
            warn(
                "On line left click for the first right click for the rest left click to end"
            )
            return

        if len(self.drawing.data[0]) > 0:
            # Get bounding box coordinates
            y_min, y_max, x_min, x_max = get_bounding_box_from_coords(
                self.drawing.data[0]
            )

            # Calculate boundaries for cropping
            y_min_bound = max([0, y_min - bounding_box])
            y_max_bound = min(
                [self.outlines.data[0].shape[0] - 1, y_max + bounding_box]
            )
            x_min_bound = max([0, x_min - bounding_box])
            x_max_bound = min(
                [self.outlines.data[0].shape[1] - 1, x_max + bounding_box]
            )

            # Convert drawing to labels
            full_line = self.drawing.to_labels(labels_shape=self.outlines.data[0].shape)
            full_line_subarray = full_line[
                y_min_bound : y_max_bound + 1, x_min_bound : x_max_bound + 1
            ].astype(bool)

            # Store current state in history
            self.history.append(
                [
                    self.outlines.data[
                        self.ui_widget.lower_change : self.ui_widget.upper_change,
                        y_min_bound + 5 : y_max_bound - 5 + 1,
                        x_min_bound + 5 : x_max_bound - 5 + 1,
                    ].copy(),
                    self.cell_mask.data[
                        self.ui_widget.lower_change : self.ui_widget.upper_change,
                        y_min_bound + 5 : y_max_bound - 5 + 1,
                        x_min_bound + 5 : x_max_bound - 5 + 1,
                    ].copy(),
                    (self.ui_widget.lower_change, self.ui_widget.upper_change),
                    (y_min_bound, y_max_bound),
                    (x_min_bound, x_max_bound),
                ]
            )

            # Loop through slices of the 3D image stack
            for i in range(self.ui_widget.lower_change, self.ui_widget.upper_change):
                outline_slice = self.outlines.data[
                    i, y_min_bound : y_max_bound + 1, x_min_bound : x_max_bound + 1
                ].astype(bool)

                # Merge drawing and outline based on mode
                if SegmenterBindings.is_deletion_mode_activated():
                    merged_subarray = ~full_line_subarray & outline_slice
                else:
                    merged_subarray = full_line_subarray | outline_slice

                # Apply padding and remove small holes
                padded_subarray = np.pad(
                    merged_subarray, pad_width=50, mode="constant", constant_values=True
                )
                padded_subarray = 255 * skimage.morphology.remove_small_holes(
                    padded_subarray, 10
                )

                # Perform watershed segmentation
                watershed_canvas = (
                    skimage.segmentation.watershed(padded_subarray, watershed_line=True)
                    == 0
                )
                watershed_canvas = watershed_canvas.astype(np.uint8) * 255

                # Crop to original size
                original_watershed = watershed_canvas[55:-55, 55:-55]

                # Update the original 3D image stack
                self.outlines.data[
                    i,
                    y_min_bound + 5 : y_max_bound - 5 + 1,
                    x_min_bound + 5 : x_max_bound - 5 + 1,
                ] = original_watershed

            # Early closure detection for polygon
        # Early closure detection for polygon
        self.outlines.refresh()
        self.update_tissue_mask()

    def remove_all_outlines_frame(self):
        """ """
        pass

    def remove_selected_outlines(self):
        """ """
        pass

    def update_tissue_mask(self):
        """ """
        coords_y, coords_x = zip(*self.drawing.data[0])
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

        current_slice = self.cell_mask.data[self.slider_pos]
        y_min, y_max = int(min(coords_y)), int(max(coords_y))
        x_min, x_max = int(min(coords_x)), int(max(coords_x))

        # Define a small neighborhood with padding
        padding = 25
        y_min_padded = max(0, y_min - padding)
        y_max_padded = min(current_slice.shape[0], y_max + padding)
        x_min_padded = max(0, x_min - padding)
        x_max_padded = min(current_slice.shape[1], x_max + padding)
        # Create a mask of the drawn area
        roi_current_slice = current_slice[
            y_min_padded:y_max_padded, x_min_padded:x_max_padded
        ]

        # Label the remaining outlines after deletion
        remaining_outlines = self.outlines.data[
            self.slider_pos, y_min_padded:y_max_padded, x_min_padded:x_max_padded
        ]
        labeled_outlines = skimage.measure.label(
            (remaining_outlines == 0), connectivity=1
        )
        removed_label = np.unique(labeled_outlines * (roi_current_slice == 255))[
            :
        ]  # Skip 0
        if SegmenterBindings.is_deletion_mode_activated():
            # Get the current cell mask slice

            # Find which label was removed by checking overlap with cell mask
            for label in removed_label:
                roi_current_slice[labeled_outlines == label] = 255

            # Update the cell mask data directly

        else:
            # Find which label was removed by checking overlap with cell mask
            # Get areas of each label (excluding 0)
            if len(removed_label) > 1:
                label_areas = []
                for label in removed_label[1:]:
                    area = np.sum(labeled_outlines == label)
                    label_areas.append((label, area))

                # Find the largest area and remove all except that one
                largest_label = max(label_areas, key=lambda x: x[1])[0]
                for label, _ in label_areas:
                    if label != largest_label:
                        roi_current_slice[labeled_outlines == label] = 0
            # Extract ROI

            # Replace updated ROI back to the full slice
        current_slice[y_min_padded:y_max_padded, x_min_padded:x_max_padded] = (
            roi_current_slice
        )

        # Refresh layers
        self.cell_mask.refresh()

    # IO AND HISTORY RELATED
    def on_export_current_labels(self, napari_viewer=None):
        """
        Export the current labels to a file.
        """
        self.create_labels()

    def create_labels(self):
        """
        Create the labels from the outlines and cell mask.
        """
        self.labels = np.array(
            [
                skimage.measure.label(self.outlines.data[i] == 0, connectivity=1)
                for i in range(self.outlines.data.shape[0])
            ]
        ) * (self.cell_mask.data == 0)

    def run_cell_tracking(self):
        """
        Save the current state of the outlines and cell mask to the public path and run the cell tracking script on the remote server.
        """
        skimage.io.imsave(
            self.public_path / "image.tif",
            self.image[:,0],
        )
        self.create_labels()

        skimage.io.imsave(
            self.public_path / "labels.tif",
            self.labels,
        )
        predictions = run_remote_tracking(
            "10.50.11.184",
            "nexton",
            os.environ.get("nexton_password"),
            "/home/nexton/Documents/trackastra-fusion/use_this_file.py",
            str(self.public_path / "image.tif"),
            str(self.public_path / "labels.tif"),
        )
        self.cell_lineage = prediction_to_cell_lineage(predictions)

    def visualize_tracking_events(self):
        if self.cell_lineage is None:
            warn("No cell lineage found, run cell tracking first")
            return
        event_dictionnary = label_events(self.cell_lineage, self.labels)
        for key, value in event_dictionnary.items():
            self.viewer.add_image(
                value,
                name=key,
                colormap=event_colors.get(key, "red"),
                blending="additive",
            )


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=str)
    args = parser.parse_args()

    # Retrieve the environment variable

    viewer = napari.Viewer()

    cell_widget = Segmenter(viewer, args.folder_path)

    viewer.window.add_dock_widget(cell_widget.ui_widget)

    napari.run()
