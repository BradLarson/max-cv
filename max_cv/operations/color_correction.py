from max.dtype import DType
from max.graph import ops, Shape, TensorType, TensorValue
from .common import assert_luminance, assert_rgb
"""Color correction operations."""

def brightness(image: TensorValue, brightness: float) -> TensorValue:
    """Adjusts the brightness of an image.

    Args:
        image: A value representing an incoming image in a graph.
        brightness: The amount by which to adjust the brightness, typically in
        the range -1.0 - 1.0.

    Returns:
        A value representing the corrected image.
    """
    assert_rgb(image)
    # The custom ops way.
    return ops.custom(
        name="brightness",
        values=[
            ops.constant(brightness, dtype=DType.float32),
            image
        ],
        out_types=[TensorType(dtype=image.dtype, shape=image.shape)],
    )[0].tensor

    # The simple way
    # return image + brightness

def gamma(image: TensorValue, gamma: float) -> TensorValue:
    """Adjusts the gamma of an image.

    Args:
        image: A value representing an incoming image in a graph.
        gamma: The amount by which to adjust the gamma, typically in
        the range 0.0 - 3.0.

    Returns:
        A value representing the corrected image.
    """
    assert_rgb(image)
    # The custom ops way.
    return ops.custom(
        name="gamma",
        values=[
            ops.constant(gamma, dtype=DType.float32),
            image
        ],
        out_types=[TensorType(dtype=image.dtype, shape=image.shape)],
    )[0].tensor

    # The simple way.
    # return ops.pow(image, gamma)

def luminance_threshold(image: TensorValue, threshold: float) -> TensorValue:
    """Sets a pixel to black if below this luminance threshold, white
    otherwise.

    Args:
        image: A value representing an incoming image in a graph.
        threshold: The luminance value to test against.

    Returns:
        A value representing the corrected image.
    """
    assert_luminance(image)
    return ops.cast(
        ops.greater(image, ops.constant(threshold, dtype=image.dtype)),
        image.dtype
    )

def rgb_to_luminance(image: TensorValue) -> TensorValue:
    """Reduces an RGB image to only its luminance channel.

    Args:
        image: A value representing an incoming image in a graph.

    Returns:
        A value representing the corrected image.
    """
    assert_rgb(image)
    image_dims = image.shape.static_dims
    image_dims[-1] = 1
    luminance_shape = Shape(image_dims)

    return ops.custom(
        name="luminance",
        values=[image],
        out_types=[TensorType(dtype=image.dtype, shape=luminance_shape)],
    )[0].tensor

def luminance_to_rgb(image: TensorValue) -> TensorValue:
    """Converts a luminance-only image back to RGB colorspace. Note: this does
    not restore colors, as a conversion to luminance is a lossy operation.

    Args:
        image: A value representing an incoming image in a graph.

    Returns:
        A value representing the corrected image.
    """
    assert_luminance(image)
    image_dims = image.shape.static_dims
    image_dims[-1] = 3
    rgb_shape = Shape(image_dims)

    return image.broadcast_to(rgb_shape)
