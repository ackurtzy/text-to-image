"""
Script to visualize sample sketches from a Quick, Draw! Dataset `.npy` file.

This script loads rasterized sketches stored as NumPy arrays from a specified file, 
randomly selects a few examples, and visualizes them using Matplotlib.

Configuration:
    - `data_path`: Path to the `.npy` file containing the rasterized sketches.

Outputs:
    A Matplotlib plot displaying a grid of 5 random sketches with their indices.
"""

import random

import numpy as np
import matplotlib.pyplot as plt

data_path = "guessed_quickdraw_data/aircraft carrier.npy"

sketches = np.load(data_path, encoding="latin1", allow_pickle=True)

# Visualize a few examples
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    index = random.randint(0, len(sketches))
    ax.imshow(sketches[index].reshape(28, 28), cmap="gray")
    ax.axis("off")
    ax.set_title(f"Image Index {index}")

fig.suptitle(f"Sample Images from {data_path.split('/')[-1]}")
plt.show()
