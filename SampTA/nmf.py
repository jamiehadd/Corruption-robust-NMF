import numpy as np
from common import relative_error
from time import time

def nmf(X_ref, X_train, max_iter, r, seed=None):
    """
    Runs the standard multiplicative updates NMF algorithm.

    Parameters:
        X_ref (np.ndarray): Data used for error measurement.
                            (This can be either the uncorrupted data or X_train itself.)
        X_train (np.ndarray): Data used for training the model.
        max_iter (int): Number of iterations.
        r (int): Target rank for the factorization.
        seed (int): Random number generator seed

    Returns:
        W (np.ndarray): Learned dictionary matrix.
        H (np.ndarray): Learned representation matrix.
        (None): For consistency of return orderings. QMU returns the mask matrix here.
        errors (list): List of relative error values computed against X_ref.
        runtime (float): Runtime of algorithm, not including relative error measurements
    """
    m, n = X_train.shape

    # Set seed for consistency across experiments
    if seed is not None:
        np.random.seed(seed)

    # Initialize factor matrices with nonnegative entries.
    W = np.abs(np.random.randn(m, r))
    H = np.abs(np.random.randn(r, n))

    errors = [relative_error(X_ref, W, H)]
    runtime = 0

    for i in range(max_iter):
        start_time = time()
        epsilon = 1e-10

        # Multiplicative update rules for standard NMF.
        W = W * ((X_train @ H.T) / (((W @ H) @ H.T) + epsilon))
        H = H * ((W.T @ X_train) / ((W.T @ (W @ H)) + epsilon))

        # Increment the runtime and calculate the relative error.
        runtime += time() - start_time
        errors.append(relative_error(X_ref, W, H))

    return W, H, None, errors, runtime
