import numpy as np
from pyanimalprocessing.plotting import plot_property

# I want you to update the EventCorrectorClass :


# You will add two layers of points with opacity 0.3 and size 20. one is dark blue (apoptosis)  the other is green. There is just an additional logic. If a point is added, you check the position (rounded) in the apoptosis_layer forapoptosis and division layer for divisions. If they interesect (meaning that there is a 1 at the pixel position) the point is a cross, otherwise it is a disc. Implement the logic cleanly


class EventCorrector:
    def __init__(
        self,
        viewer,
        apoptosis_layer,  # Napari image layer for intersection checks & property display
        divisions_layer,  # Napari image layer for intersection checks & property display
        coords_layer,  # Napari points layer for displaying original properties
        lagrangian_coords_layer,  # Napari points layer
    ):
        self.viewer = viewer
        self.apoptosis_layer = apoptosis_layer
        self.divisions_layer = divisions_layer
        self.coords_layer = coords_layer
        self.lagrangian_coords_layer = lagrangian_coords_layer
        self.self_roi_number = self.create_roi_number()

        # Determine ndim for points layers from the source image layers
        apoptosis_ndim = getattr(self.apoptosis_layer, "ndim", 3)
        divisions_ndim = getattr(self.divisions_layer, "ndim", 3)

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

        # Connect event handlers for data changes on the new points layers
        self.corrected_apoptosis_pts_layer.events.data.connect(
            self._on_corrected_apoptosis_data_change
        )
        self.corrected_divisions_pts_layer.events.data.connect(
            self._on_corrected_divisions_data_change
        )

        self.display_layers()

    def display_layers(self):
        if self.apoptosis_layer and self.coords_layer:
            plot_property(
                self.apoptosis_layer,
                self.coords_layer,
                self.viewer,
                property_name="apoptosis",
            )
        if self.divisions_layer and self.coords_layer:
            plot_property(
                self.divisions_layer,
                self.coords_layer,
                self.viewer,
                property_name="divisions",
            )
        if self.lagrangian_coords_layer and len(self.self_roi_number) > 0:
            plot_property(
                self.self_roi_number,
                self.lagrangian_coords_layer,
                self.viewer,
                property_name="lagrangian_coords",
            )

    def create_roi_number(self):
        if (
            self.lagrangian_coords_layer is not None
            and self.lagrangian_coords_layer.data is not None
            and len(self.lagrangian_coords_layer.data) > 0
        ):
            return np.ones(len(self.lagrangian_coords_layer.data))
        return np.array([])

    def _on_corrected_apoptosis_data_change(self, event=None):
        self._update_point_symbols(
            self.corrected_apoptosis_pts_layer, self.apoptosis_layer
        )

    def _on_corrected_divisions_data_change(self, event=None):
        self._update_point_symbols(
            self.corrected_divisions_pts_layer, self.divisions_layer
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
            try:
                point_voxel_coords_float = source_image_layer_for_check.world_to_data(
                    point_world_coords
                )
            except AttributeError:
                point_voxel_coords_float = point_world_coords

            rounded_voxel_coords = np.round(point_voxel_coords_float).astype(int)

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
                    if image_data_np[tuple(rounded_voxel_coords)] == 1:
                        symbol_to_set = "cross"
                except IndexError:
                    pass

            new_symbols.append(symbol_to_set)

        points_layer_to_update.symbol = new_symbols
        points_layer_to_update.refresh()
