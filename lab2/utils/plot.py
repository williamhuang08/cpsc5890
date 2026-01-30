import matplotlib.pyplot as plt
import numpy as np

def plot_3d_positions(positions, show=True, equal_aspect=True, label=None):
    """
    Plot a 3D trajectory.

    Args:
        positions: (T, 3) array-like of [x, y, z]
        show: whether to call plt.show()
        equal_aspect: keep xyz scale equal
        label: optional legend label
    """
    positions = np.asarray(positions)

    assert positions.ndim == 2 and positions.shape[1] == 3, \
        "positions must be of shape (T, 3)"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        marker="o",
        markersize=2,
        label=label
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if label is not None:
        ax.legend()

    if show:
        plt.show()

    return fig, ax
