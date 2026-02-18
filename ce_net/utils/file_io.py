#!/usr/bin/env python3

# External
import os
import numpy as np
import numpy.typing as npt


def save_poses(file_path: str, xyz_poses: dict[int, npt.NDArray[np.float64]]):
    """
    Saves 4x4 transformation matrices to a file.

    Args:
        file_path (str): The path to the file where poses will be saved.
        xyz_poses (dict): A dictionary where keys are indices and values are 4x4 numpy arrays representing transformation matrices.
    Returns:
        None
    """
    with open(file_path, "w") as file:
        for idx, matrix_4x4 in xyz_poses.items():
            flattened_matrix = matrix_4x4.flatten()
            line = f"{idx} " + " ".join(map(str, flattened_matrix)) + "\n"
            file.write(line)


def read_bin_file(file_path, dtype, shape=None):
    """
    Reads a .bin file and reshapes the data according to the provided shape.

    Args:
        file_path (str): The path to the .bin file.
        dtype (data-type): The data type of the file content (e.g., np.float32, np.int16).
        shape (tuple, optional): The desired shape of the output array. If None, the data is returned as a 1D array.

    Returns:
        np.ndarray: The data read from the .bin file, reshaped according to the provided shape.
    """
    data = np.fromfile(file_path, dtype=dtype)
    if shape:
        return data.reshape(shape)
    return data
