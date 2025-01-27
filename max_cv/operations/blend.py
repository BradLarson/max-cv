from max.dtype import DType
from max.graph import ops, TensorType, TensorValue
from .common import assert_rgb
"""Two-image blends."""

def add_blend(
    background_image: TensorValue,
    foreground_image: TensorValue,
) -> TensorValue:
    """Performs an additive blend between a background and foreground image.

    Returns:
        A value representing the blended image.
    """
    assert_rgb(background_image)
    assert_rgb(foreground_image)
    return ops.custom(
        name="blend",
        values=[
            ops.constant(1.0, dtype=DType.float32),
            background_image,
            foreground_image
        ],
        out_types=[
            TensorType(dtype=background_image.dtype, shape=background_image.shape)
        ],
        parameters={
            "blend_mode": "add"
        }
    )[0].tensor


def dissolve_blend(
    background_image: TensorValue,
    foreground_image: TensorValue,
    intensity: float
) -> TensorValue:
    """Performs a dissolve blend between a background and foreground image.

    Args:
        intensity: How strongly to blend the foreground image above the
        background. Range is 0.0 - 1.0, with 0.0 being entirely background and
        1.0 being entirely foreground.

    Returns:
        A value representing the blended image.
    """
    assert_rgb(background_image)
    assert_rgb(foreground_image)
    return ops.custom(
        name="blend",
        values=[
            ops.constant(intensity, dtype=DType.float32),
            background_image,
            foreground_image
        ],
        out_types=[
            TensorType(dtype=background_image.dtype, shape=background_image.shape)
        ],
        parameters={
            "blend_mode": "dissolve"
        }
    )[0].tensor


def multiply_blend(
    background_image: TensorValue,
    foreground_image: TensorValue,
) -> TensorValue:
    """Performs a multiply blend between a background and foreground image.

    Returns:
        A value representing the blended image.
    """
    assert_rgb(background_image)
    assert_rgb(foreground_image)
    return ops.custom(
        name="blend",
        values=[
            ops.constant(1.0, dtype=DType.float32),
            background_image,
            foreground_image
        ],
        out_types=[
            TensorType(dtype=background_image.dtype, shape=background_image.shape)
        ],
        parameters={
            "blend_mode": "multiply"
        }
    )[0].tensor
