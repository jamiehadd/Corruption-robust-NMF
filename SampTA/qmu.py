from common import relative_error
import numpy as np


def qmu(D_tilde, D, max_iter, r, q, seed=None):
    '''
    Runs the Quantile Multiplicative Updates (QMU) algorithm.

    Parameters:
        D_tilde (np.ndarray): Reference (uncorrupted) data used for error measurement.
                              If testing on uncorrupted data, set D_tilde = D.
        D (np.ndarray): Input data (possibly corrupted) used for training the model.
        max_iter (int): Number of iterations to run.
        r (int): Target rank for the factorization.
        q (float): Quantile threshold for masking (typically set to 1 - corruption_rate).

    Returns:
        W (np.ndarray): Learned dictionary matrix.
        H (np.ndarray): Learned representation matrix.
        M (np.ndarray): Final masking matrix.
        errors (list): List of relative error values computed with respect to D_tilde.
    '''
    m, n = D.shape
    # Initialize factor matrices with nonnegative entries.
    if seed is not None:
        np.random.seed(seed)
    W = np.abs(np.random.randn(m, r))
    H = np.abs(np.random.randn(r, n))

    errors = []
    errors.append(relative_error(D_tilde, W, H))

    for i in range(max_iter):
        # Construct the quantile mask
        M = quantile_mask(D, W, H, q)

        epsilon = 1e-10

        # Update rules for W and H
        W = W * (( (M * D) @ H.T ) / ( ((M * (W @ H)) @ H.T) + epsilon ))
        H = H * (( W.T @ (M * D) ) / ( (W.T @ (M * (W @ H))) + epsilon ))
        errors.append(relative_error(D_tilde, W, H))

    return W, H, M, errors


def quantile_mask(X, W, H, q):
    '''
    Constructs a binary quantile mask M based on the error matrix.

    Parameters:
        X (np.ndarray): Data matrix (the training data).
        W (np.ndarray): Current dictionary matrix.
        H (np.ndarray): Current representation matrix.
        q (float): Quantile threshold (a number between 0 and 1).

    Returns:
        M (np.ndarray): Binary mask of the same shape as X.
    '''
    # Compute the residual error matrix.
    E = np.abs(X - np.dot(W, H))

    # Compute the q-quantile threshold.
    threshold = np.quantile(E, q)

    # Create mask: 1 for entries with error <= threshold, 0 otherwise.
    M = (E <= threshold).astype(np.float64)

    return M
