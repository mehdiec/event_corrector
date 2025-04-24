import pickle
from warnings import warn
from pathlib import Path
import napari
from napari.utils.notifications import show_info
import numpy as np
import skimage.morphology
from skimage.draw import line
from tqdm import tqdm
import zarr
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
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
from tracking import (
    label_events,
    relabel_image,
    run_remote_tracking,
    prediction_to_cell_lineage,
)
from utils import (
    masks_to_outlines,
    plot_subgraph,
    get_bounding_box_from_coords,
    get_bounding_box_from_labels
)
from binding import SegmenterBindings
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import skimage.io
import time
from functools import wraps

from scipy.sparse import coo_matrix
from skimage.measure import label as cc_label


def timing_decorator(func):
    """Decorator that prints the execution time of a function.

    Parameters
    ----------
    func : callable
        The function to be timed

    Returns
    -------
    wrapper : callable
        The wrapped function that prints timing information
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result

    return wrapper


@timing_decorator
def stitch3D(masks, stitch_threshold=0.25):
    """stitch 2D masks into 3D volume with stitch_threshold on IOU"""
    mmax = masks[0].max()
    empty = 0

    for i in tqdm(range(len(masks) - 1)):
        iou = _intersection_over_union(masks[i + 1], masks[i])[1:, 1:]
        if not iou.size and empty == 0:
            masks[i + 1] = masks[i + 1]
            mmax = masks[i + 1].max()
        elif not iou.size and not empty == 0:
            icount = masks[i + 1].max()
            istitch = np.arange(mmax + 1, mmax + icount + 1, 1, int)
            mmax += icount
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
        else:
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1) == 0.0)[0]
            istitch[ino] = np.arange(mmax + 1, mmax + len(ino) + 1, 1, int)
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
            empty = 1

    return masks


@timing_decorator
def _intersection_over_union(masks_true, masks_pred):
    """intersection over union of all mask pairs

    Parameters
    ------------

    masks_true: ND-array, int
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]

    ------------
    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix.

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

@timing_decorator
def _label_overlap(x, y):
    """Fast overlap computation using sparse matrices."""
    x = x.ravel()
    y = y.ravel()

    # Create sparse co-occurrence matrix
    overlap = coo_matrix((np.ones_like(x), (x, y))).toarray()
    return overlap


# @jit(nopython=True)
# def _label_overlap(x, y):
#     """fast function to get pixel overlaps between masks in x and y

#     Parameters
#     ------------

#     x: ND-array, int
#         where 0=NO masks; 1,2... are mask labels
#     y: ND-array, int
#         where 0=NO masks; 1,2... are mask labels

#     Returns
#     ------------

#     overlap: ND-array, int
#         matrix of pixel overlaps of size [x.max()+1, y.max()+1]

#     """
#     # put label arrays into standard form then flatten them
#     #     x = (utils.format_labels(x)).ravel()
#     #     y = (utils.format_labels(y)).ravel()
#     x = x.ravel()
#     y = y.ravel()

#     # preallocate a 'contact map' matrix
#     overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)

#     # loop over the labels in x and add to the corresponding
#     # overlap entry. If label A in x and label B in y share P
#     # pixels, then the resulting overlap is P
#     # len(x)=len(y), the number of pixels in the whole image
#     for i in range(len(x)):
#         overlap[x[i], y[i]] += 1
#     return overlap


event_colors = {
    "divisions": "green",
    "delamination": "magenta",
    "new_cells": "cyan",
    "frauds": "red",
    "past_frauds": "yellow",
    "future_frauds": "orange",
}


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, fig):
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


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


class SegmenterUI(QWidget):
    def __init__(self, viewer: napari.Viewer, path_D2) -> None:
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

        self.fig, self.ax = plt.subplots()
        self.canvas_widget = None

        # Activate drawing and history layers

        self.viewer.dims.set_point(0, 0)

        # Initialize sliders and text boxes
        max_range = self.viewer.layers["raw"].data.shape[0]
        layout = QVBoxLayout(self)
        self.slider_layout = QHBoxLayout()
        text_layout = QHBoxLayout()
        self.plotting_layout = QVBoxLayout(self)

        self.draw_parameter_group_box = QGroupBox("Drawing Parameters")
        self.cell_tracking_group_box = QGroupBox("Run Cell Tracking")
        self.tracking_event_group_box = QGroupBox("Display Tracking event")
        self.export_results_group_box = QGroupBox("Export Results")
        self.graph_group_box = QGroupBox("Graph")
        self.folder_mode_group_box = QGroupBox("Segmentation type")

        self.cell_traking_box_layout = QVBoxLayout(self.cell_tracking_group_box)
        self.tracking_box_layout = QVBoxLayout(self.tracking_event_group_box)
        self.general_draw_parameter = QVBoxLayout(self.draw_parameter_group_box)
        self.export_results_layout = QVBoxLayout(self.export_results_group_box)
        self.graph_group = QVBoxLayout(self.graph_group_box)
        self.folder_mode_layout = QVBoxLayout(self.folder_mode_group_box)

        self.plotting_group_box = QGroupBox("Time Evolution Parameters")

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
        self.plotting_layout.addWidget(self.plotting_group_box)
        self.lower_slider.valueChanged.connect(self.update_text_lower)
        self.upper_slider.valueChanged.connect(self.update_text_upper)
        self.lower_text.textChanged.connect(self.update_slider_lower)
        self.upper_text.textChanged.connect(self.update_slider_upper)
        self.general_draw_parameter.addLayout(text_layout)
        self.general_draw_parameter.addLayout(self.plotting_layout)
        self.slider_enabled = False

        # Depth controls how many nodes away from the selected node to display in the graph
        self.depth_text = self.init_text_box(str(2))
        self.graph_group.addWidget(QLabel("Graph depth (nodes):"))
        self.graph_group.addWidget(self.depth_text)

        # Time depth controls how many frames forward/backward to display in the graph
        self.time_depth_text = self.init_text_box(str(2))
        self.graph_group.addWidget(QLabel("Time depth (frames):"))
        self.graph_group.addWidget(self.time_depth_text)
        relation_options = ["both", "successors", "predecessors"]
        self.relation_text = QComboBox()
        for option in relation_options:
            self.relation_text.addItem(option)
        self.graph_group.addWidget(self.relation_text)
        self.plot_future_edges_checkbox = QCheckBox("Plot future edges")
        self.graph_group.addWidget(self.plot_future_edges_checkbox)
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

        # Widget to choose a specific folder for segmentation
        self.choose_folder_type = QComboBox()
        self.folder_mode_layout.addWidget(self.choose_folder_type)
        layout.addWidget(self.folder_mode_group_box)
        
        self.current_folder = QLabel("")
        layout.addWidget(self.current_folder)

        # Subfolders of the animal.image.D2 available
        self.load_subfolders(path_D2)
        self.choose_folder_type.currentTextChanged.connect(self.on_selection_changed)
        
        # self.cell_traking_box_layout.addLayout(self.cell_traking_box_layout)
        layout.addWidget(self.draw_parameter_group_box)
        layout.addWidget(self.cell_tracking_group_box)
        layout.addWidget(self.tracking_event_group_box)
        layout.addWidget(self.graph_group_box)
        self.setLayout(layout)

        self.upper_bound = 0
        self.lower_bound = 0
    
    def load_subfolders(self, path):
        """Load subfolders to put them in the combo box"""

        sub_keys = list(path.keys())

        self.choose_folder_type.clear()
        self.choose_folder_type.addItems(sub_keys)

        if "label" in sub_keys:
            index = self.choose_folder_type.findText("label")
            if index >= 0:
                self.choose_folder_type.setCurrentIndex(index)
                self.current_folder.setText("Currently working on label")

    def on_selection_changed(self, folder_name):
        """Current folder changed"""
        self.current_folder.setText(f"Currently working on {folder_name}")

    def clear_figure(self):
        """Creates a canvas widget if not already created, draws the plot, and optionally saves it."""
        self.fig, self.ax = plt.subplots()
        if self.canvas_widget is not None:
            self.plotting_layout.removeWidget(self.canvas_widget)
            self.canvas_widget.close()
            self.canvas_widget = None
        # Update and draw plot
        self.ax.grid(True, which="both")
        self.ax.legend(loc="best")
        self.fig.canvas.draw_idle()

    def plot(self, graph, start_node):
        """Creates a canvas widget if not already created, clears previous plot, draws the new plot, and optionally saves it."""

        # Clear any existing plots
        depth = int(self.depth_text.text())
        time_depth = int(self.time_depth_text.text())
        relation = self.relation_text.currentText()
        plot_future_edges = self.plot_future_edges_checkbox.isChecked()
        if self.canvas_widget is None:
            self.canvas = MplCanvas(plt.figure(figsize=(12, 8)))
            # Create toolbar, passing canvas as first parameter, parent (self, the MainWindow) as second.
            toolbar = NavigationToolbar(self.canvas, self)
            layout = QVBoxLayout()
            layout.addWidget(toolbar)
            layout.addWidget(self.canvas)

            # Create a placeholder widget to hold our toolbar and canvas.
            self.canvas_widget = QWidget()
            self.canvas_widget.setLayout(layout)
            self.graph_group.addWidget(self.canvas_widget)
            self.canvas_widget.show()

        # Call the plot_subgraph function to generate the subgraph and get positions
        self.canvas.axes.cla()
        self.canvas.figure, self.canvas.axes = plot_subgraph(
            graph, start_node, depth, time_depth, relation, plot_future_edges
        )
        self.canvas.show()

        # Trigger the canvas_widget to update and redraw
        self.canvas.draw()

        # Refresh the graph group widget and adjust size
        self.graph_group_box.adjustSize()
        self.graph_group_box.update()

        # Force layout update
        if self.canvas_widget:
            self.canvas_widget.updateGeometry()
            self.canvas_widget.adjustSize()

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
        self.history_manager = HistoryManager()

        self.public_path = Path(os.environ.get("path_public"))
        self.image = self.animal.IMAGE.D2.raw
        self.path_D2 = self.animal.IMAGE.D2
        self.folder_type = "label"  #default folder
        
        self.viewer: napari.Viewer = viewer
        self.raw = self.viewer.add_image(
            np.array(self.image),
            name="raw",
            channel_axis=1,
        )


        self.viewer = viewer
        self.drawing_is_active = None
        self.history = []
        self.shape = "Free Hand"

        

        self.ui_widget = SegmenterUI(viewer, self.path_D2)

        self.slider_pos = int(self.viewer.dims.point[0])
        
        self.edition_mode = "automatic"
        self.complete_segmentation = {}

        self.upper_bound = 0
        self.lower_bound = 0
        self._connect_widget()
        

        

        self.change_segmentation_folder()

        # TO CHECK (changement de self.labels -> check if what is below is still doing what it is supposed to)
        self.cell_lineage = None
        
        # TODO: maybe check the origin of the tracking and give it a different name depending on the origin and the type of labels tracked
        if os.path.exists(Path(animal_path) / "cell_lineage_matlab.pkl"):
            with open(Path(animal_path) / "cell_lineage_matlab.pkl", "rb") as f:
                self.cell_lineage = pickle.load(f)
            relabeled_stuff = relabel_image(self.labels, self.cell_lineage)
            self.viewer.add_labels(relabeled_stuff, name="relabeled")
        # if os.path.exists(Path(animal_path) / "cell_lineage.pkl"):
        #     with open(Path(animal_path) / "cell_lineage.pkl", "rb") as f:
        #         self.cell_lineage = pickle.load(f)

        elif os.path.exists(Path(animal_path) / "pred.pkl"):
            with open(Path(animal_path) / "pred.pkl", "rb") as f:
                predictions = pickle.load(f)
            self.cell_lineage = prediction_to_cell_lineage(predictions, self.labels[:])
            with open(Path(animal_path) / "cell_lineage.pkl", "wb") as f:
                pickle.dump(self.cell_lineage, f)
            # exit(0)
            relabeled_stuff = relabel_image(self.labels, self.cell_lineage)
            self.viewer.add_labels(relabeled_stuff, name="relabeled")

        #
        #     self.tissue_mask = skimage.io.imread(Path(animal_path) / "labels.tif")
        # except:
        # Create tissue mask by dilating labels to make cells touch, preserving time dimension

        self.visualize_tracking_events()
     


    def _connect_widget(self):
        # Bind callback methods to mouse and key events
        self.viewer.mouse_move_callbacks.append(self.segmenting)
        self.viewer.mouse_drag_callbacks.append(self.toggle_segmenting)
        self.viewer.mouse_drag_callbacks.append(self.segmenting)
        self.viewer.mouse_drag_callbacks.append(self.remove_label)
        self.viewer.mouse_drag_callbacks.append(self.display_graph)

        # Key bindings for various actions
        self.viewer.bind_key(
            SegmenterBindings.cancel_drawing, self.clear_drawing, overwrite=True
        )
        self.viewer.bind_key(
            SegmenterBindings.save_state, self.save_current_segmentation, overwrite=True
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
        self.viewer.bind_key("Control-Z", self.perform_undo)
        self.viewer.bind_key("Control-Y", self.perform_redo)
        self.viewer.bind_key(
            SegmenterBindings.change_correcting_mode, self.switch_edition_mode, overwrite=True
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

        self.ui_widget.export_button.clicked.connect(self.save_current_segmentation)

        self.ui_widget.choose_folder_type.currentIndexChanged.connect(self.change_segmentation_folder)

    def ensure_backup_exists(self):
        """
        Create a back up of the labels if not existing
        """
        if f"{self.folder_type}_backup" not in self.path_D2:
            print("No back-up found, creating one ...")
            self.animal.IMAGE.D2.create_dataset(
                name=f"{self.folder_type}_backup",
                shape=self.labels.shape,
                dtype="uint16",
                chunks=(1, *self.labels.shape[1:]),
            )
        backup_label = self.path_D2[f"{self.folder_type}_backup"]
        for t in tqdm(range(self.labels.shape[0]), desc="Checking/generating back up"):
            if np.any(backup_label[t]):
                continue
            backup_label[t] = self.labels[t].copy()

    def ensure_skeleton_exists(self):
        """
        Get the outlines saved or create them if not existing
        """
        if f"{self.folder_type}_skeleton" not in self.path_D2:
            print(f"{self.folder_type} skeleton not found in Zarr — generating outlines...")
            self.path_D2.create_dataset(
                name=f"{self.folder_type}_skeleton",
                shape=self.labels.shape,
                dtype="uint8",
                chunks=(1, *self.labels.shape[1:]),
            )
        self.skeleton_store = self.path_D2[f"{self.folder_type}_skeleton"]
        for t in tqdm(range(self.labels.shape[0]), desc="Checking/generating skeleton"):
            if np.any(self.skeleton_store[t]):
                continue
            skeleton = masks_to_outlines(self.labels[t]).astype(np.uint8)
            self.skeleton_store[t] = skeleton
        

    def change_segmentation_folder(self):
        """
        Change the current folder being segmented
        """
        # Change folder_type based on the current selection and clear existing layers
        self.folder_type = self.ui_widget.choose_folder_type.currentText()
        print(f"current zarr: {self.folder_type}")

        # Remove any existing layers related to the selected folder_type and outlines
        if self.folder_type in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[self.folder_type])
        outlines_name = f"outlines_{self.folder_type}"
        if outlines_name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[outlines_name])
        draw_name = f"draw_{self.folder_type}"
        if draw_name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[draw_name])

        # Load the labels associated with the selected folder_type
        self.labels_original = self.animal.IMAGE.D2[self.folder_type]
        self.labels = self.animal.IMAGE.D2[self.folder_type][:].copy()
        self.labels_layer = self.viewer.add_labels(self.labels, name=self.folder_type)

        # Ensure the backup exists before proceeding
        self.ensure_backup_exists()

        # Check if skeleton exists and load it if available
        self.ensure_skeleton_exists()
        skeleton_zarr = self.skeleton_store[:].copy()  # Retrieve the skeleton data
        self.outlines_layer = self.viewer.add_labels(skeleton_zarr, name=outlines_name)

        self.drawing = self.viewer.add_shapes(
            name=f"draw_{self.folder_type}", blending="additive", shape_type="path", edge_width=2
        )
        
        

        # Refresh both the labels and outlines layers to update the viewer
        self.labels_layer.refresh()
        self.outlines_layer.refresh()

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
        self.drawing_is_active = False

    def perform_undo(self, viewer):
        state = self.history_manager.undo()
        if state:
            self.apply_state(state, undo=True)
            return
        print("stop nothing to undo")
        show_info("Nothing to undo")

    def perform_redo(self, viewer):
        state = self.history_manager.redo()
        if state:
            self.apply_state(state, undo=False)
            return
        print("stop nothing to redo")
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
        self.masks = self.labels_layer.data

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

        # If the removing mode (shift + right click) is activated
        if SegmenterBindings.is_deletion_mode_activated():
            # Drawing a circle around the click for it to be bigger and touched both labels
            if event.button == 2:
                x, y = event.position[1:]
                radius = 3  
                num_points = 20  

                theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
                circle_points = [(x + radius * np.cos(t), y + radius * np.sin(t)) for t in theta]
                self.segmenting_path += circle_points
                self.drawing.data = [self.segmenting_path]
                self.drawing_is_active = False
                self.drawing.refresh()
                self.labels_layer.refresh()
                self.outlines_layer.refresh()
                self.handle_sequential_mode()
                self.clear_drawing()

        elif self.drawing_is_active:
            if self.shape == "Free Hand":
                if self.drawing_is_active:
                    # This draws the line
                    self.segmenting_path += [event.position[1:]]
                    self.drawing.data = [self.segmenting_path]
                if not self.drawing.shape_type == "path":
                    self.drawing.shape_type = "path"
                # Set edge width to 2 pixels for thicker path
                self.drawing.edge_width = 2
            elif self.shape == "Line":
                # This draws the line
                if event.button == 1:
                    self.segmenting_path += [event.position[1:]]
                    self.drawing.data = [self.segmenting_path]
                    # if not self.drawing.shape_type == "line":
                    if not self.drawing.shape_type == "path":
                        self.drawing.shape_type = "path"
                    # Set edge width to 2 pixels for thicker path
                    self.drawing.edge_width = 2

            self.drawing.refresh()
            self.labels_layer.refresh()
            self.outlines_layer.refresh()

    def handle_sequential_mode(self, viewer=None, bounding_box=50, padding=0.2):
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

        self.update_outline(bounding_box)

    @timing_decorator
    def update_outline(self, bounding_box):
        if len(self.drawing.data[0]) > 0:

            # If the removing mode (shift + right click) is activated
            if SegmenterBindings.is_deletion_mode_activated():

                coords_arr = np.array(self.drawing.data[0], dtype=int)  
                ys = coords_arr[:, 0]
                xs = coords_arr[:, 1]
                shape = self.labels_layer.data[0].shape
                h, w = shape
                valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
                ys_valid = ys[valid]
                xs_valid = xs[valid]

                for i in range(self.ui_widget.lower_change, self.ui_widget.upper_change):

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
                    
                    offset_y = y_min 
                    offset_x = x_min
                    local_ys = ys - offset_y + padding
                    local_xs = xs - offset_x + padding


                    new_slice = self.labels_layer.data[i][y_min:y_max+1, x_min:x_max+1].copy()
                    # Padding with background to avoid problems at the border of the bounding box
                    new_slice_padded = np.pad(new_slice, pad_width=padding, mode="constant",constant_values = 0)

                    # Every pixel with the first label get the relabeled with the second
                    new_slice_padded[new_slice_padded == top2_labels[0]] = top2_labels[1]
                    self.labels_layer.data[i][y_min:y_max+1, x_min:x_max+1]=new_slice_padded[padding:-padding, padding:-padding]

                    # Patching for outline context
                    outline_borders = 2
                    y0 = min(0, y_min - outline_borders)
                    y1 = max(self.labels_layer.data[i].shape[0], y_max + 1 + outline_borders)
                    x0 = min(0, x_min - outline_borders)
                    x1 = max(self.labels_layer.data[i].shape[1], x_max + 1 + outline_borders)

                    # Outlines with more context
                    new_outline_with_context = masks_to_outlines(self.labels_layer.data[i][y0:y1, x0:x1])
                    self.outlines_layer.data[i][y0:y1, x0:x1] = new_outline_with_context[y0:y1, x0:x1]

                    after = self.labels_layer.data[
                            i,
                            y_min : y_max + 1,
                            x_min : x_max + 1
                        ].copy()
                    
                    self.history_manager.add_state(
                        0, (self.slider_pos, x_min, y_min, x_max, y_max), before, after
                    )
            
            else:
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
                
                for i in range(self.ui_widget.lower_change, self.ui_widget.upper_change):

                    label_values = self.labels_layer.data[i][ys_valid, xs_valid]
                  
                    label_values_unique = np.unique(label_values)

                    # Bounding box is defined by the drawing and the labels touched (except background)
                    y_min_draw, y_max_draw = ys_valid.min(), ys_valid.max()
                    x_min_draw, x_max_draw = xs_valid.min(), xs_valid.max()
                    if label_values_unique[label_values_unique!=0].size !=0:
                        y_min_labels, y_max_labels, x_min_labels, x_max_labels=get_bounding_box_from_labels(
                                self.labels_layer.data[i],
                                label_values_unique[label_values_unique!=0]
                            )
                        y_min, y_max = min(y_min_draw, y_min_labels), max(y_max_draw, y_max_labels)
                        x_min, x_max = min(x_min_draw, x_min_labels), max(x_max_draw, x_max_labels)
                    else:
                        y_min, y_max = y_min_draw, y_max_draw
                        x_min, x_max = x_min_draw, x_max_draw

                    before = self.labels_layer.data[
                        i,
                        y_min : y_max + 1,
                        x_min : x_max + 1
                    ].copy()

                    padding = 2

                    valid = (ys >= y_min) & (ys <= y_max+1) & (xs >= x_min) & (xs <= x_max+1)
                    ys = ys[valid]
                    xs = xs[valid]
                    
                    offset_y = y_min 
                    offset_x = x_min
                    local_ys = ys - offset_y + padding
                    local_xs = xs - offset_x + padding

                    new_slice = self.labels_layer.data[i][y_min:y_max+1, x_min:x_max+1].copy()
                    # Padding with background to avoid problems at the border of the bounding box
                    new_slice_padded = np.pad(new_slice, pad_width=padding, mode="constant",constant_values = 0)
                    # Get a label id unused
                    new_id = self.labels_layer.data[i].max() + 1

                    # For each label touched, if it is entirely cut by the drawing -> 1 new label on only one side of the drawing, the other is untouched
                    for label in label_values_unique:
                        region  = (new_slice_padded == label)
                        barrier = np.zeros_like(region)
                        barrier[local_ys, local_xs] = True
                        mask_cut = region & (~barrier)
                        comps = cc_label(mask_cut, connectivity=1)
                        if comps.max() < 2:
                            print("No cut detected")
                        # Relabelling new cell, only if size > 1 pix (to deal with solo error pixel)
                        else:
                            comps2 = [lab for lab in np.unique(comps) if lab >= 2]
                            for c in comps2:
                                comp_mask = (comps == c)
                                comp_size = comp_mask.sum()
                                if comp_size > 1:
                                    new_slice_padded[comp_mask] = new_id
                                    new_id += 1
                                    break  
                            else:
                                print("No cut detected")
                                               
                    self.labels_layer.data[i][y_min:y_max+1, x_min:x_max+1]= new_slice_padded[padding:-padding, padding:-padding]
                    # Patching for outline context
                    outline_borders = 2
                    y0 = min(0, y_min - outline_borders)
                    y1 = max(self.labels_layer.data[i].shape[0], y_max + 1 + outline_borders)
                    x0 = min(0, x_min - outline_borders)
                    x1 = max(self.labels_layer.data[i].shape[1], x_max + 1 + outline_borders)

                    # Outlines with more context
                    new_outline_with_context = masks_to_outlines(self.labels_layer.data[i][y0:y1, x0:x1])
                    self.outlines_layer.data[i][y0:y1, x0:x1] = new_outline_with_context[y0:y1, x0:x1]
                    

                    after = self.labels_layer.data[
                            i,
                            y_min : y_max + 1,
                            x_min : x_max + 1
                        ].copy()
                    self.history_manager.add_state(
                        0, (self.slider_pos, x_min, y_min, x_max, y_max), before, after
                    )
        
        self.masks = self.labels_layer.data

        # Early closure detection for polygon
        self.outlines_layer.refresh()
        self.labels_layer.refresh()
        pass

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
        self.labels_original = self.labels_layer.data
        
    def save_current_segmentation(self):
        """
        Save the current state of the labels and the outlines
        """
        self.labels_original[:] = self.labels_layer.data
        self.skeleton_store[:] = self.outlines_layer.data
        show_info(f"{self.folder_type} segmentation exported")

    def run_cell_tracking(self):
        """
        Save the current state of the outlines and cell mask to the public path and run the cell tracking script.
        """
        skimage.io.imsave(
            self.public_path / "image.tif",
            self.image[:, 0],
        )
        self.create_labels()

        skimage.io.imsave(
            self.public_path / "labels.tif",
            self.labels_original[:],
        )

        # Run tracking locally if we are nexton
        if os.environ.get("USER") == "nexton":
            import subprocess

            command = f"/home/nexton/miniforge-pypy3/envs/trackastra/bin/python /home/nexton/Documents/trackastra-fusion/use_this_file.py {str('/Volumes/u934/equipe_bellaiche/public/image.tif')} {str('/Volumes/u934/equipe_bellaiche/public/labels.tif')}"
            subprocess.run(command, shell=True, check=True)

            # Load predictions from the output file
            import pickle

            with open(self.public_path / "pred.pkl", "rb") as f:
                predictions = pickle.load(f)
        else:
            predictions = run_remote_tracking(
                "10.50.11.184",
                "nexton",
                os.environ.get("nexton_password"),
                "/home/nexton/Documents/trackastra-fusion/use_this_file.py",
                str("/Volumes/u934/equipe_bellaiche/public/image.tif"),
                str("/Volumes/u934/equipe_bellaiche/public/labels.tif"),
            )
        self.cell_lineage = prediction_to_cell_lineage(predictions, self.labels[:])

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

    def display_graph(self, viewer, event):
        """
        Handle mouse press events to delete labels with Ctrl + left click.
        """
        # Check if Ctrl is pressed and the left mouse button was clicked
        if self.cell_lineage is None:
            warn("No cell lineage found, run cell tracking first")
            return
        print("mouse down")
        dragged = False
        yield
        # Handle mouse move event
        while event.type == "mouse_move":
            dragged = True
            yield

        # Handle mouse release event after dragging
        if dragged:
            print("drag end")
            return

        print("clicked!")

        self.slider_pos = int(self.viewer.dims.point[0])
        frame = self.slider_pos
        print(f"Current frame: {frame}")

        if (
            not QApplication.instance().keyboardModifiers() & Qt.ControlModifier
            and event.button == 1
        ):
            coords = list(
                map(int, viewer.cursor.position[1:])
            )  # Convert to integer coordinates, skip z if necessary
            print(f"Click coordinates: {coords}")

            if len(coords) == 2:  # Adjust if your data is 2D
                # Clear the entire figure, not just the axes
                # Create new axes

                label_value = self.labels_layer.data[
                    self.slider_pos, coords[0], coords[1]
                ]
                if (frame, label_value) in self.cell_lineage:
                    print(f"Selected label value: {label_value}")
                    # self.prepare_and_display_plot((frame, label_value))

                    self.ui_widget.fig.canvas.draw_idle()
                    self.ui_widget.plot(self.cell_lineage, (frame, label_value))

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
                    y_min, y_max, x_min, x_max = get_bounding_box_from_coords(coords)
                    print(
                        f"Bounding box: y_min={y_min}, y_max={y_max}, x_min={x_min}, x_max={x_max}"
                    )

                    # Store state before modification
                    before = self.labels_layer.data[
                        self.slider_pos, y_min : y_max, x_min : x_max
                    ].copy()

                    # Update the data array directly
                    current_slice = self.labels_layer.data[self.slider_pos]
                    current_slice[label_mask] = 0
                    self.labels_layer.data[self.slider_pos] = current_slice

                    # Store state after modification
                    after = self.labels_layer.data[
                        self.slider_pos, y_min : y_max, x_min : x_max
                    ].copy()
                    print("Label removed successfully")
                    show_info("Label removed successfully")

                    self.history_manager.add_state(
                        label_value, (frame, x_min, y_min, x_max, y_max), before, after
                    )
                    self.masks = self.labels_layer.data
                    self.labels_layer.refresh()
                    # Patching for outline context
                    outline_borders = 2
                    y0 = min(0, y_min - outline_borders)
                    y1 = max(self.labels_layer.data[self.slider_pos].shape[0], y_max + 1 + outline_borders)
                    x0 = min(0, x_min - outline_borders)
                    x1 = max(self.labels_layer.data[self.slider_pos].shape[1], x_max + 1 + outline_borders)

                    # Outlines with more context
                    new_outline_with_context = masks_to_outlines(self.labels_layer.data[self.slider_pos][y0:y1, x0:x1])
                    self.outlines_layer.data[self.slider_pos][y0:y1, x0:x1] = new_outline_with_context[y0:y1, x0:x1]

                    self.outlines_layer.refresh()
                    print("History updated and display refreshed")

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
