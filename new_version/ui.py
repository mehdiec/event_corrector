import pickle
from warnings import warn
from pathlib import Path
import napari
from napari.utils.notifications import show_info, show_error
import numpy as np
import skimage.morphology
from skimage.draw import line
from tqdm import tqdm
import zarr
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QSpinBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QDockWidget,
    QGridLayout,
    QDialog
)
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import skimage.io
import time
from functools import wraps

from scipy.sparse import coo_matrix
from skimage.measure import label as cc_label
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QDialog, QVBoxLayout, QPushButton
from segmenter import Segmenter


class CorrectionUI(QWidget):
    def __init__(self) -> None:
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

        self.slider_pos = 0

        self._init_widgets()

    def _init_widgets(self):
        # Initialize sliders and text boxes

        layout = QVBoxLayout(self)

        self.export_results_group_box = QGroupBox("Export Results")
        self.export_results_layout = QVBoxLayout(self.export_results_group_box)
        self.init_export_widget()

        self.folder_mode_group_box = QGroupBox("Segmentation type")
        self.folder_mode_layout = QVBoxLayout(self.folder_mode_group_box)
        self.init_folder_widget()

        self.remove_holes_group_box = QGroupBox("Remove holes")
        self.remove_holes_layout = QVBoxLayout(self.remove_holes_group_box)
        self.init_remove_holes_widget()
        
        self.draw_parameter_group_box = QGroupBox("Drawing Parameters")
        self.general_draw_parameter = QVBoxLayout(self.draw_parameter_group_box)
        self.init_drawing_parameters_widget()

        self.cell_tracking_group_box = QGroupBox("Run Cell Tracking")
        self.cell_traking_box_layout = QVBoxLayout(self.cell_tracking_group_box)
        self.create_tracking_widget()

        self.tracking_event_group_box = QGroupBox("Display Tracking event")
        self.tracking_box_layout = QVBoxLayout(self.tracking_event_group_box)
        self.display_cell_tracking_event_widget()

        self.graph_group_box = QGroupBox("Graph")
        self.graph_group = QVBoxLayout(self.graph_group_box)

        # Add the Export Results layout to the main layout
        layout.addWidget(self.folder_mode_group_box)
        layout.addWidget(self.export_results_group_box)
        layout.addWidget(self.remove_holes_group_box)
        layout.addWidget(self.draw_parameter_group_box)
        layout.addWidget(self.cell_tracking_group_box)
        layout.addWidget(self.tracking_event_group_box)
        layout.addWidget(self.graph_group_box)

        self.setLayout(layout)
    
        
    
    def init_export_widget(self):
        # Initialize and configure the Export Segmentation button
        self.export_button = QPushButton("Export Segmentation")
        self.export_results_layout.addWidget(self.export_button)
    
    def init_folder_widget(self):
        # Widget to choose a specific folder for segmentation
        self.choose_folder_type = QComboBox()
        self.choose_folder_type.addItem("Choose segmentation type")
        placeholder_index = self.choose_folder_type.findText("Choose segmentation type")
        self.choose_folder_type.model().item(placeholder_index).setEnabled(False)
        self.folder_mode_layout.addWidget(self.choose_folder_type)
    
    def init_remove_holes_widget(self):        
        # Button and spinbox to remove small holes 
        self.spinbox_holes_size = QSpinBox()
        self.spinbox_holes_size.setMinimum(0)
        self.spinbox_holes_size.setValue(0)
        self.clean_holes_button= QPushButton(f"Remove holes of size {self.spinbox_holes_size.value()} pix (or less)")
        self.spinbox_holes_size.valueChanged.connect(self.update_size_holes)
        self.remove_holes_layout.addWidget(self.spinbox_holes_size)
        self.remove_holes_layout.addWidget(self.clean_holes_button)

    def init_drawing_parameters_widget(self):
        # Drawing parameters
        # # Sliders to select frames
        # self.slider_layout = QHBoxLayout()
        # self.slider_text_layout = QHBoxLayout()
        
        # self.lower_slider = QSlider(Qt.Horizontal)
        # self.slider_layout.addWidget(QLabel("Lower:"))
        # self.slider_layout.addWidget(self.lower_slider)
        # self.upper_slider = QSlider(Qt.Horizontal)
        # self.slider_layout.addWidget(QLabel("Upper:"))
        # self.slider_layout.addWidget(self.upper_slider)
        # self.general_draw_parameter.addLayout(self.slider_layout)

        # self.lower_text = self.init_text_box(str(self.slider_pos))
        # self.slider_text_layout.addWidget(self.lower_text)
        # self.upper_text = self.init_text_box(str(self.slider_pos + 1))
        # self.slider_text_layout.addWidget(self.upper_text)

        # self.lower_slider.valueChanged.connect(self.update_text_lower)
        # self.upper_slider.valueChanged.connect(self.update_text_upper)
        # self.lower_text.textChanged.connect(self.update_slider_lower)
        # self.upper_text.textChanged.connect(self.update_slider_upper)
        # self.general_draw_parameter.addLayout(self.slider_text_layout)

        # # Add a checkbox for marking segmentation as complete
        # self.checkbox_multiple_modif = QCheckBox("multiple modification")
        # self.general_draw_parameter.addWidget(self.checkbox_multiple_modif)
        # Further interface configurations
        self.possible_shapes = ["Free Hand", "Line", "Remove Segmentation"]
        self.shape = "Free Hand"
        self.combo_box_shape = QComboBox()
        for mode in self.possible_shapes:
            self.combo_box_shape.addItem(mode)
        self.general_draw_parameter.addWidget(self.combo_box_shape)
        # Add a checkbox for marking segmentation as complete
        # self.checkbox = QCheckBox("Completed Segmentation")
        # self.layout_complete_seg = QVBoxLayout()
        # self.layout_complete_seg.addWidget(self.checkbox)
        # self.general_draw_parameter.addLayout(self.layout_complete_seg)

        # Additional setup (cell tracking and display)
        # self._show_widget_cell_tracking()

        #Graph 
        # Depth controls how many nodes away from the selected node to display in the graph
        # self.depth_text = self.init_text_box(str(2))
        # self.graph_group.addWidget(QLabel("Graph depth (nodes):"))
        # self.graph_group.addWidget(self.depth_text)
        # # Time depth controls how many frames forward/backward to display in the graph
        # self.time_depth_text = self.init_text_box(str(2))
        # self.graph_group.addWidget(QLabel("Time depth (frames):"))
        # self.graph_group.addWidget(self.time_depth_text)
        # relation_options = ["both", "successors", "predecessors"]
        # self.relation_text = QComboBox()
        # for option in relation_options:
        #     self.relation_text.addItem(option)
        # self.graph_group.addWidget(self.relation_text)
        # self.plot_future_edges_checkbox = QCheckBox("Plot future edges")
        # self.graph_group.addWidget(self.plot_future_edges_checkbox)
    
        

    def set_available_folders(self, sub_keys):

        self.choose_folder_type.blockSignals(True)

        self.choose_folder_type.addItems(sub_keys)

        self.choose_folder_type.setCurrentIndex(0)
        self.choose_folder_type.blockSignals(False)

        if "label" in sub_keys:
            print("setting to labels")
            index = self.choose_folder_type.findText("label")
            print(index)
            self.choose_folder_type.setCurrentIndex(index)
    
    def get_current_folder(self):
        return self.choose_folder_type.currentText()

    def update_size_holes(self):
        self.clean_holes_button.setText(f"Remove holes of size {self.spinbox_holes_size.value()} pix (or less)")        

    def create_tracking_widget(self):
        self.button_cell_tracking = QPushButton("Run cell tracking")
        self.cell_traking_box_layout.addWidget(self.button_cell_tracking)

    def display_cell_tracking_event_widget(self):
        self.button_cell_event = QPushButton("Display tracking event")
        self.tracking_box_layout.addWidget(self.button_cell_event)
        self.button_auto_correct = QPushButton("Auto-correction")
        self.tracking_box_layout.addWidget(self.button_auto_correct)

    def switch_drawing_mode(self):
        idx = self.possible_shapes.index(self.shape)
        # passe à l’indice suivant (avec wrap-around)
        self.shape = self.possible_shapes[(idx + 1) % len(self.possible_shapes)]
        # mets à jour la combo
        self.combo_box_shape.setCurrentText(self.shape)
    
    # def init_slider(self, max_range, name):
    #     slider = QSlider(Qt.Horizontal)
    #     slider.setRange(0, max_range - 1)
    #     slider.setValue(self.slider_pos)
    #     self.slider_layout.addWidget(QLabel(name))
    #     self.slider_layout.addWidget(slider)
    #     return slider

    # def init_text_box(self, initial_text):
    #     text_box = QLineEdit()
    #     text_box.setText(initial_text)
    #     return text_box
    
    # def init_frames_slider(self, max_range):
    #     self.lower_slider.setRange(0, max_range - 1)
    #     self.lower_slider.setValue(self.slider_pos)
    #     self.upper_slider.setRange(0, max_range - 1)
    #     self.upper_slider.setValue(self.slider_pos+1)

    # def update_text_lower(self):
    #     new_value = min(self.lower_slider.value(), self.upper_slider.value() - 1)
    #     self.lower_text.setText(str(new_value))
    #     self.lower_slider.setValue(new_value)

    # def update_slider_lower(self):
    #     text_value = self.lower_text.text()
    #     if text_value.isdigit():
    #         new_value = min(int(text_value), self.upper_slider.value() - 1)
    #         # if self.lower_slider.value() != new_value:
    #         self.lower_slider.setValue(new_value)
    #         self.lower_text.setText(str(new_value))

    # def update_text_upper(self):
    #     new_value = max(self.upper_slider.value(), self.lower_slider.value()+ 1)
    #     self.upper_text.setText(str(new_value))
    #     self.upper_slider.setValue(new_value)

    # def update_slider_upper(self):
    #     text_value = self.upper_text.text()
    #     if text_value.isdigit():
    #         new_value = max(int(text_value), self.lower_slider.value() + 1)
    #         self.upper_slider.setValue(new_value)
    #         self.upper_text.setText(str(text_value))


class AutoCorrectDialog(QDialog):
    def __init__(self, viewer, parent=None):
        """
        viewer: napari Viewer instance to embed (reused)
        """
        super().__init__(parent)
        self.viewer = viewer
        self._qt_window = self.viewer.window._qt_window
        self._qt_window.setParent(self)
        self._qt_window.setWindowFlags(Qt.Widget)
        self.first_show = True

        menu_bar = self._qt_window.menuBar()
        if menu_bar:
            menu_bar.hide()
        # Initially hide dock widgets (controls)
        for dock in self._qt_window.findChildren(QDockWidget):
            dock.hide()

        self.title_widget = QWidget()
        self.title_grid = QGridLayout()
        self.title_grid.setContentsMargins(0, 0, 0, 0)
        self.title_grid.setColumnStretch(0, 1)
        self.title_grid.setColumnStretch(1, 1)
        self.lbl_left = QLabel("Potential Fraud")
        self.lbl_left.setAlignment(Qt.AlignCenter)
        self.lbl_right = QLabel("Correction Suggested")
        self.lbl_right.setAlignment(Qt.AlignCenter)
        self.title_grid.addWidget(self.lbl_left, 0, 0)
        self.title_grid.addWidget(self.lbl_right, 0, 1)
        self.title_widget.setLayout(self.title_grid)

         # Bottom status label under slider
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)

        # Buttons
        self.btn_reject = QPushButton("Refuse")
        self.btn_draw   = QPushButton("Redraw")
        self.btn_accept = QPushButton("Accept correction")
        self.btn_reject.clicked.connect(self.on_reject)
        self.btn_draw.clicked.connect(self.on_redraw)
        self.btn_accept.clicked.connect(self.on_accept)

        # Layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_reject)
        btn_layout.addWidget(self.btn_draw)
        btn_layout.addWidget(self.btn_accept)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_widget)
        main_layout.addWidget(self._qt_window)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)
        self.setWindowTitle("AutoCorrection Napari")
        self.resize(800, 600)

        # Data storage
        self.frauds = []
        self.current_index = 0
        self.labels_layer = None
        self.outlines_layer = None

    def load_frauds(self, frauds_list):
        self.frauds = frauds_list
        self.current_index = 0
        self._show_current()

    def _show_current(self):
        # Clear layers
        # for layer in list(self.viewer.layers):
        #     self.viewer.layers.remove(layer)

        raw, labels, labels_corrected, fraud, outlines, outlines_corrected, t_min, t_max, t = self.frauds[self.current_index]
        channel_in_raw = raw.shape[1]
        if self.first_show:
            
            self.first_show = False

            self.raw_left = self.viewer.add_image(np.array(raw), name="raw_left", channel_axis = 1)
            self.labels_layer_left= self.viewer.add_labels(labels, name="labels_left", opacity=0.4)
            self.outlines_layer_left  = self.viewer.add_labels(outlines, name="outlines_left")

            self.raw_right = self.viewer.add_image(np.array(raw), name="raw_right", channel_axis = 1)
            self.labels_layer_right= self.viewer.add_labels(labels_corrected, name="labels_corrected", opacity=0.4)
            self.outlines_layer_right  = self.viewer.add_labels(outlines_corrected, name="outlines_corrected")

            self.fraud_layer = self.viewer.add_labels((fraud>0).astype(np.uint8),
                                                    name="Fake_div",
                                                    colormap={0:(0,0,0,0),1:"red"},
                                                    opacity=1.0)
            self.fraud_layer.contour = 3

            self.viewer.grid.enabled = True
            self.viewer.grid.stride  = -2 -channel_in_raw
            self.viewer.grid.shape   = (1,2)

        else:
            # mises à jour rapides des data arrays
            self.raw_left[0].data = raw[:,0]
            self.raw_left[1].data = raw[:,1]
            self.labels_layer_left.data = labels
            self.outlines_layer_left.data   = outlines

            self.raw_right[0].data = raw[:,0]
            self.raw_right[1].data = raw[:,1]
            self.labels_layer_right.data = labels_corrected
            self.outlines_layer_right.data   = outlines_corrected

            self.fraud_layer.data  = (fraud>0).astype(np.uint8)
        
        self.fraud_layer.contour = 3
        self.status_label.setText(f"Displaying frames {t_min} to {t_max}")
        self.lbl_left.setText(f"Potential fake division {self.current_index}/{len(self.frauds)}")
        
        # Mark layers to show in grid
        self.viewer.dims.set_point(0,t)




    def on_accept(self):
        print(f"Accepted fraud #{self.current_index}")
        self._advance()

    def on_reject(self):
        print(f"Rejected fraud #{self.current_index}")
        self._advance()

    def on_redraw(self):
        print(f"Redrawing fraud #{self.current_index}")
        for dock in self._qt_window.findChildren(QDockWidget):
            title = dock.windowTitle()
            if title == "layer list":
                dock.show()

        if self.labels_layer_left and self.outlines_layer_left:
            self.segmenter_instance = Segmenter(self.viewer, self.labels_layer_left, self.outlines_layer_left)



    def _advance(self):
        self.current_index += 1
        if self.current_index < len(self.frauds):
            for dock in self._qt_window.findChildren(QDockWidget):
                title = dock.windowTitle()
                if (title != "frames") and (title != "titles"):
                    dock.hide()
            for layer in list(self.viewer.layers):
                layer.visible = True
            self._show_current()
        else:
            print("No more frauds, closing dialog.")
            self.accept()


class TrackingChoiceDialog(QDialog):
    """
    Boîte de dialogue pour gérer la présence d'un prédiction existante.
    Retourne:
      - "reuse"   si l'utilisateur veut réutiliser le fichier existant
      - "rerun"   si l'utilisateur veut ré-exécuter et écraser
      - "cancel"  s'il annule
    """
    def __init__(self, parent=None, pred_path=""):
        super().__init__(parent)
        self.setWindowTitle("Tracking already exists")
        self.setWindowModality(Qt.ApplicationModal)

        # Message principal
        label = QLabel(
            f"Le fichier de tracking existe déjà à :\n{pred_path}"
        )
        label.setWordWrap(True)

        # Boutons
        btn_rerun  = QPushButton("Re-run")
        btn_cancel = QPushButton("Annuler")

        btn_rerun.clicked.connect(self._rerun)
        btn_cancel.clicked.connect(self._cancel)

        # Layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_rerun)
        btn_layout.addWidget(btn_cancel)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(label)
        main_layout.addLayout(btn_layout)

        self.choice = None

    def _rerun(self):
        self.choice = "rerun"
        self.accept()   # ferme le dialog avec Accepted

    def _cancel(self):
        self.choice = "cancel"
        self.reject()