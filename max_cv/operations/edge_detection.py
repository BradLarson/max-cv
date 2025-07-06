from max.dtype import DType
from max.graph import ops, TensorType, TensorValue, DeviceRef
from max.driver import Device, CPU
from .common import assert_luminance

"""Edge detection operations."""


def sobel_edge_detection(
    device: Device, image: TensorValue, strength: float = 1.0
) -> TensorValue:
    """Performs Sobel edge detection, with edges in white.

    Args:
        image: A value representing an incoming image in a graph.
        strength: A multiplier for the edge strength, with a default of 1.0.

    Returns:
        A value representing the corrected image.
    """
    # TODO: Support inverting colorspace.
    assert_luminance(image)
    dref = DeviceRef.from_device(device)
    return ops.custom(
        name="sobel",
        device=dref,
        values=[
            ops.constant(
                strength, dtype=DType.float32, device=DeviceRef.from_device(CPU())
            ),
            image,
        ],
        out_types=[TensorType(dtype=image.dtype, shape=image.shape, device=dref)],
    )[0].tensor
