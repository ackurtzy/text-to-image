"""
Module to process Quick, Draw! NDJSON files by converting vector images into rasterized images.
"""

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cairocffi as cairo
import json


def load_vector_images(ndjson_file, max_drawings=1000):
    """
    Load vector images from Quick, Draw! NDJSON files. It only loads recognized images for quality.

    Parameters:
    - ndjson_file: Path to the NDJSON file
    - max_drawings: Number of drawings to load (optional)

    Returns:
    - vector_images: List of stroke vectors (filtered for classified images)
    """
    vector_images = []
    with open(ndjson_file, "r") as f:
        for i, line in enumerate(f):
            if i >= max_drawings:
                break
            data = json.loads(line)
            if data["recognized"]:  # Filter for classified images
                strokes = data["drawing"]
                vector_image = [np.array(stroke) for stroke in strokes]
                vector_images.append(vector_image)
    return vector_images


def vector_to_raster(
    vector_images,
    side=28,
    line_diameter=16,
    padding=16,
    bg_color=(0, 0, 0),
    fg_color=(1, 1, 1),
):
    """
    Converts a list of vector images into rasterized images.

    This function takes a list of vector images, defined as sequences of strokes,
    and rasterizes them into images of the specified size and properties. The
    rasterization process accounts for line diameter, padding, and color configuration.

    Args:
        vector_images (list): A list of vector images, where each vector image
            is a list of strokes. Each stroke is represented as a tuple of
            two arrays (x-coordinates and y-coordinates).
        side (int, optional): The side length (width and height) of the output
            raster images in pixels. Defaults to 28.
        line_diameter (int, optional): The diameter of the lines used to render
            the strokes, relative to the original 256x256 image. Defaults to 16.
        padding (int, optional): The amount of padding (in pixels) around the
            strokes, relative to the original 256x256 image. Defaults to 16.
        bg_color (tuple, optional): A 3-tuple specifying the background color
            in RGB format, where each value is between 0 and 1. Defaults to (0, 0, 0).
        fg_color (tuple, optional): A 3-tuple specifying the foreground (line)
            color in RGB format, where each value is between 0 and 1. Defaults to (1, 1, 1).

    Returns:
        list: A list of rasterized images as 1D NumPy arrays. Each rasterized image
            corresponds to an input vector image.
    """

    original_side = 256.0

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2.0 + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2.0, total_padding / 2.0)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()

        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.0
        offset = offset.reshape(-1, 1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])  # Extract pixel data
        # raster_image = raster_image.reshape((side, side))  # Reshape to 2D array
        raster_images.append(raster_image)

    return raster_images


def convert_and_save(input_path, output_path, max_drawings):
    """
    Loads vector images, rasterizes them, and saves as a NumPy array.

    This function loads vector images from a specified input file, converts them into
    raster images using the `vector_to_raster` function, and saves the resulting
    images as a NumPy array to the specified output file.

    Args:
        input_path (str): Path to the input file containing vector image data.
        output_path (str): Path to the output file where the rasterized images
            will be saved as a NumPy `.npy` file.
        max_drawings (int): Maximum number of vector images to load and process.

    Returns:
        None
    """
    vector_images = load_vector_images(input_path, max_drawings=max_drawings)
    raster_images = vector_to_raster(
        vector_images, side=28, line_diameter=16, padding=16
    )
    full_array = np.array(raster_images)
    np.save(output_path, full_array, allow_pickle=True)
