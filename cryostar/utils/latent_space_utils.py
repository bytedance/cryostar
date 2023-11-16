# This file contains code modified from cryodrgn/analysis.py 
# available at https://github.com/ml-struct-bio/cryodrgn/blob/main/cryodrgn/analysis.py
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import warnings
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning)
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = np


# silencing umap deprecation warning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import umap


def run_umap(z: np.ndarray, **kwargs) -> Tuple[np.ndarray, umap.UMAP]:
    reducer = umap.UMAP(**kwargs)
    z_embedded = reducer.fit_transform(z)
    return z_embedded, reducer


def run_pca(z: np.ndarray) -> Tuple[np.ndarray, PCA]:
    pca = PCA(z.shape[1])
    pca.fit(z)
    # print("Explained variance ratio:")
    # print(pca.explained_variance_ratio_)
    pc = pca.transform(z)
    return pc, pca


def get_pc_traj(
    pca: PCA,
    zdim: int,
    numpoints: int,
    dim: int,
    start: Optional[float] = 5,
    end: Optional[float] = 95,
    percentiles: Optional[np.ndarray] = None,
) -> npt.NDArray[np.float32]:
    """
    Create trajectory along specified principal component

    Inputs:
        pca: sklearn PCA object from run_pca
        zdim (int)
        numpoints (int): number of points between @start and @end
        dim (int): PC dimension for the trajectory (1-based index)
        start (float): Value of PC{dim} to start trajectory
        end (float): Value of PC{dim} to stop trajectory
        percentiles (np.array or None): Define percentile array instead of np.linspace(start,stop,numpoints)

    Returns:
        np.array (numpoints x zdim) of z values along PC
    """
    if percentiles is not None:
        assert len(percentiles) == numpoints
    traj_pca = np.zeros((numpoints, zdim))
    if percentiles is not None:
        traj_pca[:, dim - 1] = percentiles
    else:
        assert start is not None
        assert end is not None
        traj_pca[:, dim - 1] = np.linspace(start, end, numpoints)
    ztraj_pca = pca.inverse_transform(traj_pca)
    return ztraj_pca


# clustering
def get_nearest_point(data: np.ndarray, query: np.ndarray) -> Tuple[npt.NDArray[np.float32], np.ndarray]:
    """
    Find closest point in @data to @query
    Return datapoint, index
    """
    ind = cdist(query, data).argmin(axis=1)
    return data[ind], ind


def cluster_kmeans(z: np.ndarray, K: int, on_data: bool = True, reorder: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster z by K means clustering
    Returns cluster labels, cluster centers
    If reorder=True, reorders clusters according to agglomerative clustering of cluster centers
    """
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0, max_iter=10)
    labels = kmeans.fit_predict(z)
    centers = kmeans.cluster_centers_

    centers_ind = None
    if on_data:
        centers, centers_ind = get_nearest_point(z, centers)

    if reorder:
        # BUG from seaborn or scipy:
        # sns.clustermap only supports data with at least 2 dim
        if z.shape[1] == 1:
            centers = np.hstack([centers, np.zeros_like(centers)])
        g = sns.clustermap(centers)
        reordered = g.dendrogram_row.reordered_ind
        centers = centers[reordered]
        if centers_ind is not None:
            centers_ind = centers_ind[reordered]
        tmp = {k: i for i, k in enumerate(reordered)}
        labels = np.array([tmp[k] for k in labels])
        if z.shape[1] == 1:
            centers = centers[:, :1]
    return labels, centers


def sample_along_pca(z: np.ndarray, pca_dim=1, num=5) -> np.ndarray:
    assert isinstance(z, np.ndarray)
    pc, pca = run_pca(z)
    start = np.percentile(pc[:, pca_dim - 1], 5)
    stop = np.percentile(pc[:, pca_dim - 1], 95)
    z_pc_traj = get_pc_traj(pca, z.shape[1], num, pca_dim, start, stop)
    point, point_id = get_nearest_point(z, z_pc_traj)
    return point, point_id
