from max.dtype import DType
from max.graph import ops, TensorType, TensorValue
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