from max.dtype import DType
from max.graph import ops, TensorType, TensorValue
from .common import assert_rgb
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
