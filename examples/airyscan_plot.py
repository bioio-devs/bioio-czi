"""Plot Airyscan detector planes and sum channel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_airyscan_planes(
    detectors: np.ndarray,
    sum_channel: np.ndarray,
    *,
    detector_label: str,
    sum_label: str,
    output: Path,
) -> None:
    """
    Plot all H detector slices in a grid plus the sum channel below.

    Parameters
    ----------
    detectors
        Array with shape (H, Y, X).
    sum_channel
        Array with shape (Y, X).
    detector_label
        Title prefix for detector panels (e.g. channel name).
    sum_label
        Title for the sum channel panel.
    output
        Path to write the PNG figure.
    """
    h_count, _, _ = detectors.shape
    cols = 8
    detector_rows = int(np.ceil(h_count / cols))

    fig = plt.figure(figsize=(2.2 * cols, 2.2 * detector_rows + 3), layout="constrained")
    grid = fig.add_gridspec(detector_rows + 1, cols, height_ratios=[1] * detector_rows + [1.4])

    vmin, vmax = np.percentile(detectors, (1, 99.5))

    for h in range(h_count):
        row, col = divmod(h, cols)
        ax = fig.add_subplot(grid[row, col])
        ax.imshow(detectors[h], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"H={h}", fontsize=8)
        ax.axis("off")

    sum_ax = fig.add_subplot(grid[detector_rows, :])
    sum_vmin, sum_vmax = np.percentile(sum_channel, (1, 99.5))
    sum_ax.imshow(sum_channel, cmap="gray", vmin=sum_vmin, vmax=sum_vmax, interpolation="nearest")
    sum_ax.set_title(sum_label, fontsize=10)
    sum_ax.axis("off")

    fig.suptitle(f"{detector_label}: {h_count} detector planes + sum", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
