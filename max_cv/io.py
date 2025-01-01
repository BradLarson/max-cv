import numpy as np
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.graph import ops, TensorValue
from pathlib import Path
from PIL import Image
from .operations import luminance_to_rgb
"""Common I/O functionality for loading, saving, and processing images."""


def load_image_into_tensor(path: Path, device: Device = CPU()) -> Tensor:
    """Loads an image from the provided path into a MAX driver Tensor, and
    moves that Tensor onto a device if needed.
    
    Args:
        path: The location of the image to load.
        device: An optional device to move the tensor onto. If unspecified,
        leaves on the CPU.

    Returns:
        A MAX driver Tensor with a UInt8 datatype containing the image in HWC format.
    """
    image = Image.open(path)
    image_array = np.asarray(image)
    return Tensor.from_numpy(image_array).to(device)
    # image_shape = image_array.shape


def normalize_image(image: TensorValue, dtype: DType) -> TensorValue:
    """Normalizes an image tensor to a 0.0 - 1.0 color range by first
    converting it into the provided floating-point datatype and then dividing
    the color channels by 255.
    
    Args:
        image: A graph TensorValue representing an input image.
        dtype: The floating-point datatype to be used in the internal graph.

    Returns:
        A graph value representing the result of the image normalization.
    """
    # TODO: Assert that the input datatype is uint8
    return ops.cast(image, dtype) / 255.0


def restore_image(tensor: TensorValue, clamp: bool = True) -> TensorValue:
    """After all the actions of the image pipeline have completed, restores
    the image to a 0-255 colorspace and places it back in a UInt8 format. If
    the inbound image is luminance-only, it is converted back to RGB
    colorspace.
    
    Args:
        tensor: A value representing the end result of the image pipeline.
        clamp: Whether to clamp the incoming color channels to a 0.0 - 1.0
        range before converting to UInt8.

    Returns:
        A graph value representing a UInt8 image tensor with color channels in
        the 0-255 range.
    """
    if clamp:
        input = ops.max(ops.min(tensor, 1.0), 0.0)
    else:
        input = tensor

    if input.shape[-1] == 1:
        input = luminance_to_rgb(input)

    return ops.cast(input * 255.0, dtype=DType.uint8)
