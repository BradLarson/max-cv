from max.dtype import DType
from max.graph import ops, TensorType, TensorValue
from .common import assert_rgb
import numpy as np
from typing import Optional

def draw_circle(
        image: TensorValue,
        radius: int,
        color: tuple,
        width: int,
        center: Optional[tuple] = None
    ) -> TensorValue:
    """Draw a circle onto the given image.

    Args:
        image: A value representing an incoming image in a graph.
        radius: The radius of the circle.
        color: The color of the circle, represented as an RGB triple, eg. (255, 0, 0) for red.
        width: The pixel width of the circle.
        center: The center point of the circle, defaults to the center of the image.
    Returns:
        A value with a circle inserted into the original value.
    """
    assert_rgb(image)
    c = center or [image.shape.static_dims[0]//2, image.shape.static_dims[1]//2]
    return ops.custom(
        name='draw_circle',
        values=[
            image,
            ops.constant(radius, dtype=DType.float32),
            ops.constant(np.array(color)/255.0, dtype=DType.float32),
            ops.constant(width, dtype=DType.float32),
            ops.constant(np.array(c), dtype=DType.float32),
        ],
        out_types=[TensorType(dtype=image.dtype, shape=image.shape)],
    )[0].tensor