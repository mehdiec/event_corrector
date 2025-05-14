import zarr
import napari
from pathlib import Path
import os
import numpy as np
from ui import CorrectionUI
from tqdm import tqdm
from functools import wraps
import time
from folder_manager import FolderManager
from tracking import CellTracker
import skimage
from controller import SegmentingController
from segmenter import Segmenter
from qtpy.QtWidgets import QMessageBox

def init(viewer: napari.Viewer, animal_path):
    animal = zarr.open(animal_path)

    folder_manager = FolderManager(animal)
    raw_image = folder_manager.get_raw_image()
    sub_keys = folder_manager.get_subkeys()
    path_public = Path(os.environ.get("path_public"))

    viewer = viewer
    ui_widget = CorrectionUI()
        
    viewer.add_image(
            np.array(raw_image),
            name="raw",
            channel_axis=1,
        )
    
    viewer.dims.set_point(0, 0)
    # ui_widget.init_frames_slider(raw_image.shape[0])
        
    def on_folder():
        current_folder = ui_widget.get_current_folder()
        if current_folder is None:
            return
        
        for layer in list(viewer.layers):
            if 'raw' not in layer.name:
                viewer.layers.remove(layer)

        folder_manager.ensure_backup_exists(current_folder)
        outlines = folder_manager.ensure_skeleton_exists(current_folder)
        labels = folder_manager.get_labels(current_folder)
        labels_layer = viewer.add_labels(labels, name=current_folder)
        outlines_layer = viewer.add_labels(outlines, name= f"outlines_{current_folder}")

        segmenter = Segmenter(viewer, labels_layer, outlines_layer)
        tracker = CellTracker(path_public, raw_image)
        ui_widget._controller = SegmentingController(viewer, ui_widget, folder_manager, segmenter, tracker)

    ui_widget.choose_folder_type.currentIndexChanged.connect(on_folder)
    ui_widget.set_available_folders(sub_keys)
    
    
    return ui_widget


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=str)
    args = parser.parse_args()

    # Retrieve the environment variable
    title = Path(args.folder_path).name
    viewer = napari.Viewer(title=title)

    cell_widget = init(viewer, args.folder_path)

    viewer.window.add_dock_widget(cell_widget)

    napari.run()