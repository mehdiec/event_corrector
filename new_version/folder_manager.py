import pickle
import time
from functools import wraps

import numpy as np
import zarr
from tqdm import tqdm
from utils_new import masks_to_outlines


def timing_decorator(func):
    """Decorator that prints the execution time of a function.

        Parameters
        ----------
        func : callable
            The function to be timed
    -
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


class FolderManager:
    def __init__(self, animal_path):
        self.animal_path = animal_path
        self.pred_path = self.animal_path / "pred.pkl"
        self.lineage_path = self.animal_path / "cell_lineage.pkl"
        self.animal = zarr.open(animal_path)
        self.image = self.animal.IMAGE.D2.raw
        self.path_D2 = self.animal.IMAGE.D2
        self.labels = None
        self.skeleton_store = None
        self.completed_store = None

    def get_subkeys(self):
        return list(self.path_D2.keys())

    def get_raw_image(self):
        return self.image

    def get_labels(self, current_folder):
        return self.path_D2[current_folder][:].copy()

    @timing_decorator
    def ensure_backup_exists(self, current_folder):
        """
        Create a back up if not existing
        """
        self.labels = self.path_D2[current_folder]
        data = self.labels[:]

        if f"{current_folder}_backup" not in self.path_D2:
            print("No back-up found, creating one ...")
            self.path_D2.create_dataset(
                name=f"{current_folder}_backup",
                shape=data.shape,
                dtype="uint16",
                chunks=(1, *data.shape[1:]),
            )
        backup_label = self.path_D2[f"{current_folder}_backup"]

        for t in tqdm(range(data.shape[0]), desc="Checking/generating back up"):
            if np.any(backup_label[t]):
                continue
            backup_label[t] = self.labels[t].copy()
        return backup_label

    @timing_decorator
    def ensure_skeleton_exists(self, current_folder):
        """
        Get the outlines saved or create them if not existing
        """
        data = self.labels[:]
        if f"{current_folder}_skeleton" not in self.path_D2:
            print(
                f"{current_folder} skeleton not found in Zarr — generating outlines..."
            )
            self.path_D2.create_dataset(
                name=f"{current_folder}_skeleton",
                shape=data.shape,
                dtype="uint8",
                chunks=(1, *data.shape[1:]),
            )

        self.skeleton_store = self.path_D2[f"{current_folder}_skeleton"]
        for t in tqdm(range(data.shape[0]), desc="Checking/generating skeleton"):
            if np.any(self.skeleton_store[t]):
                continue
            skeleton = masks_to_outlines(data[t]).astype(np.uint8)
            self.skeleton_store[t] = skeleton
        return self.skeleton_store[:].copy()

    def save_segmentation(self, labels_layer, outlines_layer):
        if (self.labels is not None) and (self.skeleton_store is not None):
            self.labels[:] = labels_layer
            self.skeleton_store[:] = outlines_layer
        else:
            print("Impossible to save")

    def check_for_pred(self, get=False):
        if not self.pred_path.exists():
            return None
        else:
            if get:
                with open(self.pred_path, "rb") as f:
                    predictions = pickle.load(f)
                return self.pred_path, predictions
            else:
                return self.pred_path

    def save_preds(self, predictions):
        print(f"Saving predictions of tracking in {self.pred_path}")
        try:
            with open(self.pred_path, "wb") as f:
                pickle.dump(predictions, f)
            print(f"Predictions saved successfully to {self.pred_path}")
        except Exception as e:
            print(f"Error saving predictions: {e}")
            raise

    def save_cell_lineage(self, cell_lineage):
        try:
            with open(self.lineage_path, "wb") as f:
                pickle.dump(cell_lineage, f)
        except Exception as e:
            print(f"Error saving predictions: {e}")
            raise
