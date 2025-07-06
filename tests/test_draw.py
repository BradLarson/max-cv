import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import TensorType, DeviceRef
import max_cv.operations as ops
from .common import generate_test_tensor, run_graph, make_graph


def test_draw(session: InferenceSession) -> None:
    device = CPU()
    input_shape = (100, 100, 3)
    image_tensor = generate_test_tensor(
        device,
        dtype=DType.float32,
        shape=input_shape,
    )
    graph = make_graph(
        "draw_circle",
        forward=lambda x: ops.draw_circle(device, x, 4, (1.0, 0.0, 0.0), 1),
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
    assert result.shape == input_shape

    input_values = image_tensor.to_numpy()
    output_values = result.to_numpy()

    assert np.all(output_values >= 0), "Output values should be non-negative"
    assert np.all(output_values <= 1), "Output values should be <= 1"

    assert not np.allclose(input_values, output_values), (
        "Output should be different from input after drawing circle"
    )

    # Find pixels that have been changed (where circle was drawn)
    changed_pixels = ~np.isclose(input_values, output_values)
    changed_pixel_count = np.sum(changed_pixels)
    assert changed_pixel_count > 0, (
        "Some pixels should have been changed by circle drawing"
    )

    # The output should still be valid image data
    assert not np.any(np.isnan(output_values)), "Output should not contain NaN values"
    assert not np.any(np.isinf(output_values)), (
        "Output should not contain infinite values"
    )
