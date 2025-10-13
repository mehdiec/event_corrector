import csv
import os

import numpy as np
from pyanimalprocessing.plotting import plot_property
import zarr

# I want you to update the EventCorrectorClass :


# You will add two layers of points with opacity 0.3 and size 20. one is dark blue (apoptosis)  the other is green. There is just an additional logic. If a point is added, you check the position (rounded) in the apoptosis_layer forapoptosis and division layer for divisions. If they interesect (meaning that there is a 1 at the pixel position) the point is a cross, otherwise it is a disc. Implement the logic cleanly


class EventCorrector:
    def __init__(self, viewer, animal_path):
        self.viewer = viewer
        self.animal = zarr.open(animal_path)
        self.animal_name = animal_path.name
        self.self_roi_number = self.create_roi_number()

        self.display_layers()

        # Determine ndim for points layers from the source image layers
        apoptosis_ndim = 3
        divisions_ndim = 3

        # Retrieve apoptosis and divisions layers
        self.apoptosis_layer = self.viewer.layers["apoptosis"]
        self.divisions_layer = self.viewer.layers["divisions"]

        # Create new points layers for corrected events
        self.corrected_apoptosis_pts_layer = self.viewer.add_points(
            None,
            ndim=apoptosis_ndim,
            name="Corrected Apoptosis",
            face_color="darkblue",
            opacity=0.3,
            size=20,
            symbol="disc",
        )
        self.corrected_divisions_pts_layer = self.viewer.add_points(
            None,
            ndim=divisions_ndim,
            name="Corrected Divisions",
            face_color="green",
            opacity=0.3,
            size=20,
            symbol="disc",
        )

        # Load previously saved points if they exist
        self._load_existing_corrections()

        # Connect event handlers for data changes on the new points layers
        self.corrected_apoptosis_pts_layer.events.data.connect(
            self._on_corrected_apoptosis_data_change
        )
        self.corrected_divisions_pts_layer.events.data.connect(
            self._on_corrected_divisions_data_change
        )
        print("EventCorrector: Event handlers connected.")  # Debug print

    def _load_existing_corrections(self):
        """Load previously saved corrections from CSV files if they exist."""
        # Load apoptosis corrections
        apoptosis_data = self._load_points_from_csv("apoptosis_correction.csv")
        if apoptosis_data is not None and len(apoptosis_data) > 0:
            self.corrected_apoptosis_pts_layer.data = apoptosis_data
            self._update_point_symbols(
                self.corrected_apoptosis_pts_layer, self.apoptosis_layer
            )
            print(
                f"EventCorrector: Loaded {len(apoptosis_data)} apoptosis corrections."
            )

        # Load divisions corrections
        divisions_data = self._load_points_from_csv("divisions_correction.csv")
        if divisions_data is not None and len(divisions_data) > 0:
            self.corrected_divisions_pts_layer.data = divisions_data
            self._update_point_symbols(
                self.corrected_divisions_pts_layer, self.divisions_layer
            )
            print(
                f"EventCorrector: Loaded {len(divisions_data)} divisions corrections."
            )

    def _load_points_from_csv(self, filename):
        """Loads 3D points data from a CSV file.

        Returns the points data as a numpy array, or None if file doesn't exist.
        """
        # Get the same path logic as used in _save_points_to_csv
        store_filesystem_path = self.animal.store.path
        parent_dir_of_store = os.path.dirname(store_filesystem_path)
        load_path = os.path.join(
            parent_dir_of_store, self.animal_name, filename
        )

        print(load_path)

        if not os.path.exists(load_path):
            return None

        points_data = []
        with open(load_path, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header

            for row in reader:
                if len(row) == 3:
                    point = [float(row[0]), float(row[1]), float(row[2])]
                    points_data.append(point)

        if points_data:
            return np.array(points_data)
        else:
            return None

    def _save_points_to_csv(self, points_layer, filename):
        """Saves 3D points data from a layer to a CSV file.

        This method is specifically designed for 3D points layers as used for
        corrected apoptosis and division events in this class.
        The CSV file is saved in the parent directory of the animal's Zarr store.
        """
        if (
            points_layer is None
            or points_layer.data is None
            or len(points_layer.data) == 0
        ):
            # Using points_layer.name if available for better logging
            layer_name = (
                points_layer.name
                if hasattr(points_layer, "name") and points_layer.name
                else "the layer"
            )
            print(f"EventCorrector: No data in {layer_name} to save to {filename}.")
            return

        if points_layer.ndim != 3:
            print(
                f"EventCorrector: Warning: Expected 3D points layer for saving to CSV, \
                  got {points_layer.ndim}D for layer '{points_layer.name}'. \
                  Skipping save for {filename}."
            )
            return

        try:
            # self.animal is a zarr group, self.animal.store is the Zarr store object,
            # and self.animal.store.path is its filesystem path (can be a file or directory).
            store_filesystem_path = self.animal.store.path
        except AttributeError:
            print(
                "EventCorrector: Critical Error: self.animal.store.path not found. \
                  Cannot determine save location for CSV. Please check animal object."
            )
            return

        # Save CSV in the same directory AS the Zarr store file/directory.
        # e.g., if store is /path/to/data/myanimal.zarr, CSV is /path/to/data/filename.csv
        # e.g., if store is /path/to/data/myanimal_zarr_dir/, CSV is /path/to/data/filename.csv
        parent_dir_of_store = os.path.dirname(store_filesystem_path)
        save_path = os.path.join(
            parent_dir_of_store, self.animal_name, filename
        )

        # Ensure the target directory (parent_dir_of_store) exists.
        # This directory should ideally always exist as it contains the Zarr store.
        try:
            os.makedirs(parent_dir_of_store, exist_ok=True)
        except OSError as e:
            print(
                f"EventCorrector: Error creating directory {parent_dir_of_store}: {e}. \
                  Cannot save {filename}."
            )
            return

        points_data = points_layer.data
        # Header for 3D points, consistent with previous use for ndim=3.
        # 't' might represent a slice index or first spatial dim like 'z'.
        header = ["t", "y", "x"]

        try:
            with open(save_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(points_data)
            layer_name = (
                points_layer.name
                if hasattr(points_layer, "name") and points_layer.name
                else "the layer"
            )
            print(
                f"EventCorrector: Saved {len(points_data)} points from {layer_name} to {save_path}"
            )
        except IOError as e:
            print(f"EventCorrector: Error writing to CSV file {save_path}: {e}")
        except Exception as e:
            print(
                f"EventCorrector: An unexpected error occurred while writing {save_path}: {e}"
            )

    def display_layers(self):

        plot_property(
            self.animal.CELL.D2.apoptosis,
            self.animal.CELL.D2.coords,
            viewer=self.viewer,
            property_name="apoptosis",
        )
        # Set colormap for apoptosis layer
        if "apoptosis" in self.viewer.layers:
            self.viewer.layers["apoptosis"].colormap = "darkblue"
            self.viewer.layers["apoptosis"].blending = "additive"

        plot_property(
            self.animal.CELL.D2.divisions,
            self.animal.CELL.D2.coords,
            viewer=self.viewer,
            property_name="divisions",
        )
        # Set colormap for divisions layer
        if "divisions" in self.viewer.layers:
            self.viewer.layers["divisions"].colormap = "green"
            self.viewer.layers["divisions"].blending = "additive"

        plot_property(
            self.self_roi_number,
            self.animal.GRID.LAGRANGIAN_PIV.GRID_PROPERTIES.lagrangian_grid,
            viewer=self.viewer,
            property_name="lagrangian_coords",
        )

    def create_roi_number(self):

        return np.ones(
            self.animal.GRID.LAGRANGIAN_PIV.GRID_PROPERTIES.lagrangian_grid.shape
        )[:, np.newaxis, ..., np.newaxis].astype(np.uint8)

    def _on_corrected_apoptosis_data_change(self, event=None):
        print(
            "EventCorrector: _on_corrected_apoptosis_data_change triggered."
        )  # Debug print
        self._update_point_symbols(
            self.corrected_apoptosis_pts_layer, self.apoptosis_layer
        )
        self._save_points_to_csv(
            self.corrected_apoptosis_pts_layer, "apoptosis_correction.csv"
        )

    def _on_corrected_divisions_data_change(self, event=None):
        print(
            "EventCorrector: _on_corrected_divisions_data_change triggered."
        )  # Debug print
        self._update_point_symbols(
            self.corrected_divisions_pts_layer, self.divisions_layer
        )
        self._save_points_to_csv(
            self.corrected_divisions_pts_layer, "divisions_correction.csv"
        )

    def _update_point_symbols(
        self, points_layer_to_update, source_image_layer_for_check
    ):
        if points_layer_to_update is None or source_image_layer_for_check is None:
            return

        points_data = points_layer_to_update.data
        if points_data is None or len(points_data) == 0:
            points_layer_to_update.symbol = []
            points_layer_to_update.refresh()
            return

        image_data_np = source_image_layer_for_check.data
        if image_data_np is None:
            return

        image_shape = image_data_np.shape
        new_symbols = []

        for point_world_coords in points_data:
            print(f"New point world coordinates: {point_world_coords}")  # Debug print
            try:
                point_voxel_coords_float = source_image_layer_for_check.world_to_data(
                    point_world_coords
                )
            except AttributeError:
                point_voxel_coords_float = point_world_coords

            rounded_voxel_coords = np.round(point_voxel_coords_float).astype(int)
            print(f"Rounded voxel coordinates: {rounded_voxel_coords}")  # Debug print

            if len(rounded_voxel_coords) > len(image_shape):
                rounded_voxel_coords = rounded_voxel_coords[-len(image_shape) :]

            valid_indices = True
            if len(rounded_voxel_coords) == len(image_shape):
                for i in range(len(rounded_voxel_coords)):
                    if not (0 <= rounded_voxel_coords[i] < image_shape[i]):
                        valid_indices = False
                        break
            else:
                valid_indices = False

            symbol_to_set = "disc"
            if valid_indices:
                try:
                    value_at_coords = image_data_np[tuple(rounded_voxel_coords)]
                    print(
                        f"Value at {rounded_voxel_coords} in source image: {value_at_coords}"
                    )  # Debug print
                    if value_at_coords == 1:
                        symbol_to_set = "cross"
                except IndexError:
                    print(f"IndexError at {rounded_voxel_coords}")  # Debug print
                    pass

            new_symbols.append(symbol_to_set)

        points_layer_to_update.symbol = new_symbols
        points_layer_to_update.refresh()
