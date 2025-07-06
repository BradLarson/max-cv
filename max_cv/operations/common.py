from max.graph import TensorValue

"""Shared helper functions for operations."""


def assert_rgb(image: TensorValue):
    """Tests that an image has RGB channels.

    Args:
        image: A value representing an incoming image in a graph.

    Raises:
        ValueError: if the input image does not have 3 color channels.
    """
    channels = image.shape.static_dims[-1]
    if channels != 3:
        raise ValueError(
            f"expected an RGB image, instead has only {channels} channel(s)"
        )


def assert_luminance(image: TensorValue):
    """Tests that an image has only a luminance channel.

    Args:
        image: A value representing an incoming image in a graph.

    Raises:
        ValueError: if the input image does not have one luminance channel.
    """
    channels = image.shape.static_dims[-1]
    if channels != 1:
        raise ValueError(
            f"expected a luminance-only image, instead has {channels} channels"
        )
