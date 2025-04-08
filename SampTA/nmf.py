import numpy as np
from common import relative_error


def nmf(X_ref, X_train, max_iter, r, seed=None):
    """
    Runs the standard multiplicative updates NMF algorithm.

    Parameters:
        X_ref (np.ndarray): Data used for error measurement.
                            (This can be either the uncorrupted data or X_train itself.)
        X_train (np.ndarray): Data used for training the model.
        max_iter (int): Number of iterations.
        r (int): Target rank for the factorization.

    Returns:
        W (np.ndarray): Learned dictionary matrix.
        H (np.ndarray): Learned representation matrix.
        errors (list): List of relative error values computed against X_ref.
    """
    m, n = X_train.shape
    # Initialize factor matrices with nonnegative entries.
    if seed is not None:
        np.random.seed(seed)
    W = np.abs(np.random.randn(m, r))
    H = np.abs(np.random.randn(r, n))

    errors = [relative_error(X_ref, W, H)]

    for i in range(max_iter):
        epsilon = 1e-10
        # Multiplicative update rules for standard NMF.
        W = W * ((X_train @ H.T) / (((W @ H) @ H.T) + epsilon))
        H = H * ((W.T @ X_train) / ((W.T @ (W @ H)) + epsilon))

        errors.append(relative_error(X_ref, W, H))

    return W, H, None, errors
