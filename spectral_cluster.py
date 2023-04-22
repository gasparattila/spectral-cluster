from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import _spectral_cluster_impl
from _spectral_cluster_impl import ClusterError
import scipy.sparse
from sklearn.cluster import k_means

__all__ = ['solve', 'ClusterError', 'generate_sbm']


def _gpu_mul(adj_matrix):
    import cupy
    import cupyx

    if scipy.sparse.issparse(adj_matrix):
        adj_matrix = cupyx.scipy.sparse.csr_matrix(adj_matrix)
        mul = adj_matrix.__mul__
    else:
        adj_matrix = cupy.asarray(adj_matrix)
        mul = adj_matrix.__matmul__
    n = adj_matrix.shape[0]
    if adj_matrix.shape != (n, n):
        raise ValueError("adj_matrix must be a square matrix")

    def op(in_vec, out_vec):
        (mul(cupy.asarray(in_vec))).get(out=out_vec)

    return n, op


def solve(adj_matrix: NDArray[np.bool_] | scipy.sparse.spmatrix,
          cluster_count=2, *, use_gpu=False, max_iter=100, tol=0.1, ncv: Optional[int] = None, debug=False
          ) -> NDArray[np.int_]:
    """
    Performs a clustering assignment for a graph using the algorithm in
    ["Spectral redemption in clustering sparse networks"](https://www.pnas.org/doi/10.1073/pnas.1312486110).

    Parameters:
        adj_matrix: The adjacency matrix of the graph.
        cluster_count: The number of clusters to find.
        use_gpu: If True, matrix multiplications are done on a GPU. Requires the `cupy` library to be installed.

    Returns:
        An array of integers in the range `1 .. (cluster_count-1)` representing the group assignment.

    Raises:
        ClusterException: If the algorithm fails.
    """
    kwargs = {'count': cluster_count - 1, 'max_iter': max_iter,
              'tol': tol, 'ncv': ncv, 'debug': debug}
    if use_gpu:
        ev = _spectral_cluster_impl.get_eigenvectors_from_op(
            *_gpu_mul(adj_matrix), **kwargs)
    else:
        if scipy.sparse.issparse(adj_matrix):
            ev = _spectral_cluster_impl.get_eigenvectors_from_sparse(
                adj_matrix, **kwargs)
        else:
            ev = _spectral_cluster_impl.get_eigenvectors_from_dense(
                adj_matrix, **kwargs)
    return k_means(ev, cluster_count)[1]


def generate_sbm(sizes: NDArray[np.int_], prob_matrix: NDArray[np.float_], *,
                 sparse=False, seed: Optional[int] = None
                 ) -> Tuple[NDArray[np.int_], NDArray[np.bool_] | scipy.sparse.csr_matrix]:
    """
    Generates an undirected graph using the Stochastic Block Model.

    Parameters:
        sizes: An array of length `n` containing the sizes of the groups.
        prob_matrix: A symmetric array of shape `(n, n)` containing the edge probabilities between groups.
        sparse: If True, a `scipy.sparse.csr_array` is returned instead of a numpy array.
        seed: An optional seed for the random number generator.

    Returns:
        labels: An array of integers in the range `0 .. n-1` containing the group assignments of vertices.
        matrix: The adjacency matrix of the graph.
    """
    labels, matrix = _spectral_cluster_impl.generate_sparse_sbm(
        np.asarray(sizes), np.asarray(prob_matrix), seed)
    if not sparse:
        matrix = matrix.todense()
    return labels, matrix
