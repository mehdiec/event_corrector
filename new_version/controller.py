
from napari import Viewer
from ui import CorrectionUI
from segmenter import Segmenter
from folder_manager import FolderManager
from tracking import CellTracker
import napari
from napari.utils.notifications import show_info
import numpy as np
from ui import AutoCorrectDialog

class SegmentingController:
    def __init__(self, viewer: Viewer, ui_widget: CorrectionUI, folder_manager: FolderManager, segmenter: Segmenter, tracker: CellTracker):
        self.viewer = viewer
        self.ui_widget = ui_widget
        self.folder_manager = folder_manager
        self.segmenter = segmenter
        self.tracker = tracker
        self.relabeled = None 
        self.event_dictionnary = None
        self.connect_widget()
        self.connect_keyboard()
        
    def connect_widget(self):
        self.ui_widget.clean_holes_button.clicked.connect(self.pushed_button_clean_holes)
        self.ui_widget.export_button.clicked.connect(self.export_segmentation)
        self.ui_widget.combo_box_shape.currentIndexChanged.connect(self.change_drawing_mode)
        self.ui_widget.button_cell_tracking.clicked.connect(self.run_tracking)
        self.ui_widget.button_cell_event.clicked.connect(self.vizualize_tracking)
        self.ui_widget.button_auto_correct.clicked.connect(self.autocorrect)

    def connect_keyboard(self):
        self.viewer.bind_key("Shift-s", self.export_segmentation, overwrite = True)
        self.viewer.bind_key("m", self.change_drawing_mode_ui, overwrite=True)

    def pushed_button_clean_holes(self):
        size_holes = self.ui_widget.spinbox_holes_size.value()+1
        self.segmenter.clean_holes(size_holes)

    def export_segmentation(self, viewer):
        print("Saving...")
        labels_layer = self.segmenter.get_labels_layer().data[:]
        outlines_layer = self.segmenter.get_outlines_layer().data[:]
        self.folder_manager.save_segmentation(labels_layer, outlines_layer)
        show_info("Segmentation exported")
    
    def change_drawing_mode_ui(self, viewer):
        self.ui_widget.switch_drawing_mode()

    def change_drawing_mode(self):
        self.segmenter.set_shape(self.ui_widget.combo_box_shape.currentText()) 
    
    def run_tracking(self):
        labels = self.segmenter.get_labels_layer().data[:]
        self.tracker.run(labels)
        show_info("Tracking done")

    def vizualize_tracking(self):

        labels = self.segmenter.get_labels_layer().data[:]
        self.relabeled, self.event_dictionnary = self.tracker.visualize_tracking_events(labels)
        self.viewer.add_labels(self.relabeled, name= "relabeled", opacity=0.5)
        event_colors = {
            "divisions": "#00FF00",
            "delamination": "magenta",
            "new_cells": "cyan",
            "frauds": "red",
            "past_frauds": "yellow",
            "future_frauds": "orange",
        }

        for key, mask in self.event_dictionnary.items():
            labels = self.viewer.add_labels(
                mask.astype(np.uint8),       
                name=f"{key}",
                colormap= {
                    0: (0, 0, 0, 0),
                    1: event_colors[key],
                },
                opacity=1.0,
            )
            labels.contour = 3
        self.segmenter.set_labels_opacity(0.5)

    
    def autocorrect(self):
        events = self.tracker.get_event().get('frauds', [])
        all_labels   = self.segmenter.get_labels_layer().data
        all_outlines = self.segmenter.get_outlines_layer().data
        raw_image = self.folder_manager.get_raw_image()
        raw_image = np.array(raw_image)

        # Create a single hidden napari Viewer once
        viewer_hidden = napari.Viewer(show=False)

        # Prepare cropped stacks for all events
        frauds_data = []
        for t, label_id in events:
            
            mask = (all_labels[t] == label_id)
            coords = np.argwhere(mask)
            if coords.size == 0:
                continue
            y_min, y_max = coords[:,0].min(), coords[:,0].max()
            x_min, x_max = coords[:,1].min(), coords[:,1].max()
            y_min, y_max = max(y_min-60, 0), min(y_max+60, all_labels.shape[1])
            x_min, x_max = max(x_min-60, 0), min(x_max+60, all_labels.shape[2])
            f_min, f_max = max(0, t-3), min(all_labels.shape[0], t+4)
            relabels = self.relabeled[f_min:f_max, y_min:y_max, x_min:x_max]
            labels = all_labels[f_min:f_max, y_min:y_max, x_min:x_max]
            outlines = all_outlines[f_min:f_max, y_min:y_max, x_min:x_max]
            label_ids_in_relabeled = self.relabeled[t, y_min:y_max, x_min:x_max][mask[y_min:y_max, x_min:x_max]]
            unique_ids, counts = np.unique(label_ids_in_relabeled, return_counts=True)

            # Ignore background (usually id=0)
            valid = unique_ids != 0
            unique_ids, counts = unique_ids[valid], counts[valid]

            # Take the most common id in the overlapping area (assume it's the same cell)
            if len(unique_ids) == 0:
                continue  # nothing found
            new_id = unique_ids[np.argmax(counts)]
            mask_stack = (relabels == new_id)
            raw = raw_image[f_min:f_max, 0,y_min:y_max, x_min:x_max]
            frauds_data.append((raw, labels, mask_stack, outlines, f_min, f_max, t - f_min))

        # Instantiate dialog once, reuse viewer_hidden
        dlg = AutoCorrectDialog(viewer_hidden, parent=self.viewer.window.qt_viewer)
        dlg.load_frauds(frauds_data)
        dlg.exec_()

    def __del__(self):
        print("SegmentingController destroyed")
    