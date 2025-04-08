import numpy as np


def relative_error(X, W, H):
    '''
    Computes the relative Frobenius norm error between the reference data X
    and its approximation W @ H.

    Parameters:
        X (np.ndarray): Reference data (can be corrupted or uncorrupted).
        W (np.ndarray): Dictionary matrix.
        H (np.ndarray): Representation matrix.

    Returns:
        float: Relative error computed as ||X - W @ H||_F^2 / ||X||_F^2.
    '''
    epsilon = 1e-10
    error_norm = np.linalg.norm(X - W @ H, 'fro')**2
    ref_norm = np.linalg.norm(X, 'fro')**2
    return error_norm / (ref_norm + epsilon)
