from max.dtype import DType
from max.graph import ops, TensorType, TensorValue
from .common import assert_luminance
"""Edge detection operations."""


def sobel_edge_detection(image: TensorValue, strength: float = 1.0) -> TensorValue:
    """Performs Sobel edge detection, with edges in white.

    Args:
        image: A value representing an incoming image in a graph.
        strength: A multiplier for the edge strength, with a default of 1.0.

    Returns:
        A value representing the corrected image.
    """
    # TODO: Support inverting colorspace.
    assert_luminance(image)
    return ops.custom(
        name="sobel",
        values=[
            ops.constant(strength, dtype=DType.float32),
            image
        ],
        out_types=[TensorType(dtype=image.dtype, shape=image.shape)],
    )[0].tensor
