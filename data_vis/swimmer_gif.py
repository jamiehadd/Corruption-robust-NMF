import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from data_gen import load_swimmer_dataset
from nmf import nmf
from qmu import qmu

# Load the dataset
D, D_tilde = load_swimmer_dataset(beta=0.05, corruption_scale=5)

# Run algorithms and get reconstruction histories
_, _, _, _, _, recs_nmf = nmf(D_tilde, D, max_iter=400, r=17, seed=42)
_, _, _, _, _, recs_qmu = qmu(D_tilde, D, max_iter=400, r=17, q=0.95, seed=42)

def make_gif(recs, out_path, total_duration=None, speedup_factor=8):
    """Builds a GIF from reconstruction history with exponential timing and clamped tail durations."""
    eps = 1e-1
    # extract and reshape column 17
    mats = [(rec[:, 17].reshape(11, 20) + eps) for rec in recs]

    # shared log-normalization
    vmin = min(M.min() for M in mats)
    vmax = max(M.max() for M in mats)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    # timing schedule
    n = len(mats)
    total_duration = total_duration or (n / 10)  # total GIF time in seconds
    raw = np.exp(-speedup_factor * np.arange(n) / (n - 1))
    secs = raw / raw.sum() * total_duration
    durations_ms = [max(1, int(s * 1000)) for s in secs]
    # clamp durations after iteration 150
    if len(durations_ms) > 150:
        clamp = durations_ms[150]
        for j in range(150, len(durations_ms)):
            durations_ms[j] = clamp

    frames = []
    for idx, M in enumerate(mats):
        # full-frame image matching data aspect ratio
        fig = plt.figure(figsize=(10, 5.5), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(M, cmap='plasma', norm=norm, aspect='equal')
        ax.axis('off')

        # iteration label starting at 0
        ax.text(
            0.18, 0.95, f"Iter {idx}",
            transform=ax.transAxes,
            fontsize=32, fontweight='bold', fontfamily='sans-serif',
            color='white', verticalalignment='top', horizontalalignment='right'
        )

        # render to RGBA buffer and extract RGB channels correctly
        buf, (w, h) = fig.canvas.print_to_buffer()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        rgb = img[:, :, :3]  # take R, G, B channels
        frames.append(rgb)
        plt.close(fig)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, frames, duration=durations_ms)
    print(f"Wrote {out_path}")

# Generate whitespace-free, exponentially-timed GIFs with viridis cmap
make_gif(recs_nmf, "gifs/nmf_reconstruction.gif", speedup_factor=8)
make_gif(recs_qmu, "gifs/qmu_reconstruction.gif", speedup_factor=8)

