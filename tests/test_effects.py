import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, DeviceRef
import max_cv.operations as ops
from .common import generate_test_tensor, run_graph, make_graph


def test_pixellate(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(
        device, dtype=DType.float32, shape=(100, 100, 3)
    )
    graph = make_graph(
        "pixellate",
        forward=lambda x: ops.pixellate(device, x, 10),
        input_types=[
            TensorType(
                image_tensor.dtype,
                shape=image_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == (100, 100, 3)

    # Verify pixellation creates block patterns
    input_values = image_tensor.to_numpy()
    output_values = result.to_numpy()

    # Verify output values are in valid range
    assert np.all(output_values >= 0), "Output values should be non-negative"
    assert np.all(output_values <= 1), "Output values should be <= 1"

    # Check that pixellation creates blocks - adjacent pixels in blocks should be identical
    block_size = 10
    for i in range(0, 100, block_size):
        for j in range(0, 100, block_size):
            # Get the block region
            block = output_values[
                i : min(i + block_size, 100), j : min(j + block_size, 100), :
            ]
            if block.size > 0:
                # All pixels in the block should be identical (or very close)
                first_pixel = block[0, 0, :]
                for c in range(3):
                    assert np.allclose(block[:, :, c], first_pixel[c], atol=1e-5), (
                        f"Block pixels should be identical in channel {c}"
                    )


def test_gaussian_blur(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(
        device, dtype=DType.float32, shape=(100, 100, 3)
    )

    graph = make_graph(
        "gaussian blur",
        forward=lambda x: ops.gaussian_blur(device, x, 3, 3.0, True),
        input_types=[
            TensorType(
                image_tensor.dtype,
                shape=image_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    # This check is a bit iffy depending on the input shape, could be off by 1
    # in some cases
    assert result.shape == (100, 100, 3)

    # Verify gaussian blur is applied
    input_values = image_tensor.to_numpy()
    output_values = result.to_numpy()

    # Verify output values are in valid range
    assert np.all(output_values >= 0), "Output values should be non-negative"
    assert np.all(output_values <= 1), "Output values should be <= 1"

    # Verify blur has been applied - output should be different from input (unless input was constant)
    input_variance = np.var(input_values)
    output_variance = np.var(output_values)

    if input_variance > 0:  # If input has variation
        assert not np.allclose(input_values, output_values), (
            "Output should be different from input after blurring"
        )
        # Blurring should generally reduce variance (smoothing effect)
        assert output_variance <= input_variance * 1.1, (
            "Blurring should reduce or maintain variance"
        )

    # Test edge preservation - center pixels should be more smoothed than edge pixels
    # This is a basic test that blur has a smoothing effect
    center_region = output_values[25:75, 25:75, :]
    center_gradients = np.gradient(center_region, axis=0).flatten()
    input_center_region = input_values[25:75, 25:75, :]
    input_center_gradients = np.gradient(input_center_region, axis=0).flatten()

    # After blur, gradients should generally be smaller (smoother)
    if np.std(input_center_gradients) > 0:
        assert (
            np.mean(np.abs(center_gradients))
            <= np.mean(np.abs(input_center_gradients)) * 1.1
        ), "Blur should reduce gradients"
