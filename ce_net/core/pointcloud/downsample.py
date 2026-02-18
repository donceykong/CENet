
import numpy as np

def random_downsample_pc(semantic_pc, max_points):
    """Downsamples a sematic pointcloud.
    Args:
        semantic_pc (numpy.ndarray): Nx4 array of point positions and semantic IDs.
        max_points (float): number of indices from semantic pc to randomly sample.

    Returns:
        semantic_pc_ds (ndarray): 
            This array is the downsampled points and labels of the semantically-labelled pc.
    """

    downsampled_indices = np.random.choice(len(semantic_pc), max_points, replace=False)
    semantic_pc_ds = np.asarray(semantic_pc)[downsampled_indices, :]

    return semantic_pc_ds