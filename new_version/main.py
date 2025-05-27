import os
import time
from functools import wraps
from pathlib import Path

import napari
import numpy as np
import skimage
import zarr
from controller import SegmentingController
from event_displayer import EventCorrector
from folder_manager import FolderManager
from qtpy.QtWidgets import QMessageBox
from segmenter import Segmenter
from tqdm import tqdm
from tracking import CellTracker
from ui import CorrectionUI


def init(viewer: napari.Viewer, animal_path):

    folder_manager = FolderManager(Path(animal_path))
    raw_image = folder_manager.get_raw_image()
    sub_keys = folder_manager.get_subkeys()

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
            if "raw" not in layer.name:
                viewer.layers.remove(layer)

        folder_manager.ensure_backup_exists(current_folder)
        outlines = folder_manager.ensure_skeleton_exists(current_folder)
        labels = folder_manager.get_labels(current_folder)
        labels_layer = viewer.add_labels(labels, name=current_folder)
        outlines_layer = viewer.add_labels(outlines, name=f"outlines_{current_folder}")

        apoptosis_layer = folder_manager.get_apoptosis()
        divisions_layer = folder_manager.get_divisions()
        coords_layer = folder_manager.get_coords()
        lagrangian_coords_layer = folder_manager.get_lagrangian_coords()

        segmenter = Segmenter(viewer, labels_layer, outlines_layer)
        tracker = CellTracker(Path(animal_path).name, raw_image)
        ui_widget._controller = SegmentingController(
            viewer, ui_widget, folder_manager, segmenter, tracker
        )

        event_corrector = EventCorrector(
            viewer,
            apoptosis_layer,
            divisions_layer,
            coords_layer,
            lagrangian_coords_layer,
        )

    ui_widget.choose_folder_type.currentIndexChanged.connect(on_folder)
    ui_widget.set_available_folders(sub_keys)

    return ui_widget


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=str)
    args = parser.parse_args()

    # Retrieve the environment variable
    title = Path(args.folder_path).name
    viewer = napari.Viewer(title=title)

    cell_widget = init(viewer, args.folder_path)

    viewer.window.add_dock_widget(cell_widget)

    napari.run()
