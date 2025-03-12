from max.dtype import DType
from max.graph import ops, TensorType, TensorValue
from .common import assert_rgb
import numpy as np
"""Stylized image effects."""


def pixellate(image: TensorValue, pixel_width: int) -> TensorValue:
    """Pixellates an image into small squares.

    Args:
        image: A value representing an incoming image in a graph.
        pixel_width: The edge length of a pixel square.

    Returns:
        A value representing the corrected image.
    """
    assert_rgb(image)
    return ops.custom(
        name="pixellate",
        values=[
            ops.constant(pixel_width, dtype=DType.int32),
            image
        ],
        out_types=[TensorType(dtype=image.dtype, shape=image.shape)],
    )[0].tensor

def gaussian_blur(image: TensorValue, kernel_size: int=3, sigma: float=1., padding: bool=False) -> TensorValue:
    """Apply a gaussian blur affect to the image.

    Args:
        image: A value representing an incoming image in a graph.
        kernel_size: The size of the blur kernel to be applied.
        sigma: Standard deviation used for computing the kernel.
        padding: Whether to pad the input image to avoid losing pixels at the edge if the image.
    
    Returns:
        A value representing the blurred image.
    """
    assert_rgb(image)
    N = kernel_size

    # generate the kernel
    ax = np.linspace(-(N - 1) / 2., (N - 1) / 2., N)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    arr =  (kernel / np.sum(kernel))

    # reshape the expected RSCF layout
    kernel = ops.constant(arr, dtype=image.dtype)
    kernel = kernel.reshape((N, N, 1, 1))
    kernel = kernel.broadcast_to((N, N, 1, 3))
    
    pad_value = (N // 2) if padding else 0

    return ops.conv2d(
        # expects NHWC layout but we only ever have 1 input image
        image.reshape([1] + image.shape.static_dims),
        kernel,
        groups=3,
        padding=[pad_value] * 4
    )[0].tensor
