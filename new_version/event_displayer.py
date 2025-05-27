import numpy as np
from pyanimalprocessing.plotting import plot_property

# I want you to update the EventCorrectorClass :


# You will add two layers of points with opacity 0.3 and size 20. one is dark blue (apoptosis)  the other is green. There is just an additional logic. If a point is added, you check the position (rounded) in the apoptosis_layer forapoptosis and division layer for divisions. If they interesect (meaning that there is a 1 at the pixel position) the point is a cross, otherwise it is a disc. Implement the logic cleanly


class EventCorrector:
    def __init__(self, viewer, animal):
        self.viewer = viewer
        self.animal = animal
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

        # Connect event handlers for data changes on the new points layers
        self.corrected_apoptosis_pts_layer.events.data.connect(
            self._on_corrected_apoptosis_data_change
        )
        self.corrected_divisions_pts_layer.events.data.connect(
            self._on_corrected_divisions_data_change
        )
        print("EventCorrector: Event handlers connected.")  # Debug print

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

    def _on_corrected_divisions_data_change(self, event=None):
        print(
            "EventCorrector: _on_corrected_divisions_data_change triggered."
        )  # Debug print
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
