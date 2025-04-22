import numpy as np
import scipy.io
import matplotlib.pylab as plt
import os

def corrupt_matrix(D_tilde, beta=0.1, corruption_scale=1e6):
    """
    Generates a corrupted data matrix D from an uncorrupted data matrix D_tilde.

    This function simulates corruption by adding noise to a fraction of the entries in D_tilde.
    The observed data is given by:
        D = D_tilde + C,
    where C is a sparse corruption matrix. A fraction beta of the entries in D_tilde
    are corrupted by adding noise drawn from the absolute value of a Gaussian distribution,
    i.e., |N(0, corruption_scale^2)|, ensuring nonnegative noise.

    Parameters:
        D_tilde (np.ndarray): The original, uncorrupted data matrix.
        beta (float): The fraction of entries to corrupt (0 < beta < 1).
        corruption_scale (float): The standard deviation of the noise, determining the magnitude of corruption.

    Returns:
        D (np.ndarray): The corrupted data matrix.
    """
    m, n = D_tilde.shape
    total_elements = m * n
    num_corrupted = int(total_elements * beta)

    # Generate nonnegative noise: absolute value of a Gaussian random variable
    noise = np.abs(corruption_scale * np.random.randn(num_corrupted))

    # Create a copy of D_tilde and flatten it for easier indexing
    D_flat = D_tilde.copy().flatten()

    # Randomly select indices to corrupt
    corrupt_indices = np.random.choice(total_elements, size=num_corrupted, replace=False)

    # Add noise to the selected entries
    D_flat[corrupt_indices] += noise

    # Reshape back to the original matrix shape
    D = D_flat.reshape(m, n)

    return D


def generate_synthetic_matrix(m, n, r, beta=0, corruption_scale=1e6):
    """
    Generates a synthetic data matrix D_tilde that is exactly factorizable as
        D_tilde = W_tilde @ H_tilde,
    where W_tilde and H_tilde have nonnegative integer entries drawn uniformly from {0, ..., 99}.

    Parameters:
        m (int): Number of rows in D_tilde.
        n (int): Number of columns in D_tilde.
        r (int): Target rank (latent dimensionality) of the factorization.
        beta (float): Proportion of matrix entries to be corrupted.
        corruption_scale (float): Size of corruptions to be added.

    Returns:
        D (np.ndarray): The corrupted data matrix.
        D_tilde (np.ndarray): The uncorrupted synthetic data matrix.
    
    """
    # Generate factor matrices using the paper's notation.
    W_tilde = np.abs(np.random.randint(0, high=100, size=(m, r)))
    H_tilde = np.abs(np.random.randint(0, high=100, size=(r, n)))

    # Form the uncorrupted data matrix D_tilde.
    D_tilde = W_tilde @ H_tilde
    D_tilde = D_tilde.astype(float)  # Convert to float to allow for future noise addition

    D = D_tilde
    if beta > 0:
        D = corrupt_matrix(D_tilde, beta, corruption_scale=corruption_scale)

    return D, D_tilde


def load_swimmer_dataset(beta=0, corruption_scale=1e6, display=False):
    """
    Loads the Swimmer dataset from the 'Swimmer.mat' file.
    Ensure 'Swimmer.mat' is in the working directory before calling.

    In the .mat file, the data is stored under the key 'X'. Optionally, two sample images
    (images 17 and 170) can be displayed.

    Parameters:
        beta (float): Proportion of matrix entries to be corrupted
        corruption_scale (float): Size of corruptions to be added
        display (bool): If True, displays two sample images from the dataset.

    Returns:
        D (np.ndarray): The data matrix from the Swimmer dataset, with corruptions.
        D_tilde (np.ndarray): The data matrix from the Swimmer dataset.
    """
    mat_path = os.path.join(os.path.dirname(__file__), "../data/Swimmer.mat")
    mat = scipy.io.loadmat(os.path.abspath(mat_path))
    D_tilde = mat['X'].astype(float)
    D = D_tilde

    if beta > 0:
        D = corrupt_matrix(D_tilde, beta, corruption_scale=corruption_scale)

    if display:
        pic17 = np.reshape(D_tilde[:, 17], (11, 20))
        pic170 = np.reshape(D_tilde[:, 170], (11, 20))

        plt.figure(figsize=[15, 6])
        plt.suptitle("Swimmer Images")

        plt.subplot(1, 2, 1)
        plt.imshow(pic17)
        plt.title("Image 17")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.imshow(pic170)
        plt.title("Image 170")
        plt.xticks([])
        plt.yticks([])

    return D, D_tilde