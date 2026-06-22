#!/usr/bin/env python3
"""
Crosstalk Calibration and Correction Library.
Provides logic for matrix generation, normalization, inversion, loading/saving profiles,
and applying correction to RGB values or image arrays.
"""

import json
import numpy as np

def compute_calibration_matrices(means_r, means_g, means_b):
    """
    Computes crosstalk matrices from average responses under R, G, B illumination.

    Parameters:
        means_r (list or np.ndarray): Mean R, G, B values under Red illumination.
        means_g (list or np.ndarray): Mean R, G, B values under Green illumination.
        means_b (list or np.ndarray): Mean R, G, B values under Blue illumination.

    Returns:
        M (np.ndarray): 3x3 raw response matrix (columns correspond to Red, Green, Blue illumination).
        M_norm (np.ndarray): 3x3 column-normalized response matrix.
        M_corr (np.ndarray): 3x3 correction matrix (inverse of M_norm).

    Raises:
        numpy.linalg.LinAlgError: If M_norm is singular and cannot be inverted.
    """
    M = np.zeros((3, 3), dtype=np.float64)
    M[:, 0] = means_r
    M[:, 1] = means_g
    M[:, 2] = means_b

    M_norm = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        diag_val = M[j, j]
        if diag_val == 0:
            diag_val = 1.0
        M_norm[:, j] = M[:, j] / diag_val

    # np.linalg.inv will raise LinAlgError if singular
    M_corr = np.linalg.inv(M_norm)

    return M, M_norm, M_corr

def apply_correction(arr, correction_matrix):
    """
    Apply the 3x3 crosstalk correction matrix to a NumPy array (or list) of RGB values.

    If the input array has an integer dtype (e.g. np.uint16 or np.uint8), the output
    is rounded (+0.5), clipped to [0, 65535], and returned as np.uint16.
    Otherwise, float values are returned.

    Parameters:
        arr (list or np.ndarray): 1D vector (3,) or multi-dimensional array ending in 3.
        correction_matrix (list or np.ndarray): 3x3 correction matrix.

    Returns:
        np.ndarray: Corrected RGB array.
    """
    arr_np = np.asarray(arr)
    corr_np = np.asarray(correction_matrix, dtype=np.float64)

    if arr_np.ndim == 1:
        corrected = np.dot(corr_np, arr_np)
        if np.issubdtype(arr_np.dtype, np.integer):
            return np.clip(corrected + 0.5, 0, 65535).astype(np.uint16)
        return corrected
    else:
        # Multi-dimensional array (e.g., shape (H, W, 3))
        # Each pixel is a row vector v. The corrected pixel is (M_corr * v^T)^T = v * M_corr^T
        corrected = np.dot(arr_np.astype(np.float32), corr_np.T)
        if np.issubdtype(arr_np.dtype, np.integer):
            return np.clip(corrected + 0.5, 0, 65535).astype(np.uint16)
        return corrected

def load_profile(filepath):
    """
    Loads a crosstalk calibration profile JSON file.

    Parameters:
        filepath (str): Path to the profile JSON file.

    Returns:
        dict: The loaded profile dictionary.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def save_profile(filepath, camera_model, speed_r, means_r, stds_r, speed_g, means_g, stds_g, speed_b, means_b, stds_b, M, M_norm, M_corr):
    """
    Saves the crosstalk calibration profile to a JSON file.
    """
    profile = {
        "camera_model": camera_model,
        "captured_data": {
            "Red": {
                "shutter_speed": speed_r,
                "means": list(means_r),
                "stds": list(stds_r)
            },
            "Green": {
                "shutter_speed": speed_g,
                "means": list(means_g),
                "stds": list(stds_g)
            },
            "Blue": {
                "shutter_speed": speed_b,
                "means": list(means_b),
                "stds": list(stds_b)
            }
        },
        "crosstalk_matrix_raw": M.tolist() if hasattr(M, "tolist") else M,
        "crosstalk_matrix_normalized": M_norm.tolist() if hasattr(M_norm, "tolist") else M_norm,
        "crosstalk_correction_matrix": M_corr.tolist() if hasattr(M_corr, "tolist") else M_corr
    }
    with open(filepath, 'w') as f:
        json.dump(profile, f, indent=4)
