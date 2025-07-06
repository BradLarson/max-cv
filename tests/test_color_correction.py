import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import TensorType, DeviceRef
import max_cv.operations as ops
from .common import generate_test_tensor, run_graph, make_graph


def test_brightness(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(
        device, dtype=DType.float32, shape=(100, 100, 3)
    )
    graph = make_graph(
        "brightness",
        forward=lambda x: ops.brightness(device, x, 0.5),
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

    input_values = image_tensor.to_numpy()
    output_values = result.to_numpy()
    expected_output = input_values + 0.5
    assert np.allclose(output_values, expected_output, atol=1e-5), (
        "Brightness should be: output = input + brightness"
    )

    test_pixels = [(10, 10), (50, 50), (90, 90)]
    for y, x in test_pixels:
        for c in range(3):
            input_val = input_values[y, x, c]
            output_val = output_values[y, x, c]
            expected_val = input_val + 0.5
            assert abs(output_val - expected_val) < 1e-5, (
                f"Pixel ({y},{x}) channel {c}: expected {expected_val}, got {output_val}"
            )


def test_gamma(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(
        device, dtype=DType.float32, shape=(100, 100, 3)
    )
    graph = make_graph(
        "gamma",
        forward=lambda x: ops.gamma(device, x, 1.5),
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

    input_values = image_tensor.to_numpy()
    output_values = result.to_numpy()
    expected_output = np.power(input_values, 1.5)
    assert np.allclose(output_values, expected_output, atol=1e-5), (
        "Gamma should be: output = input^gamma"
    )

    test_pixels = [(10, 10), (50, 50), (90, 90)]
    for y, x in test_pixels:
        for c in range(3):
            input_val = input_values[y, x, c]
            output_val = output_values[y, x, c]
            expected_val = input_val**1.5
            assert abs(output_val - expected_val) < 1e-5, (
                f"Pixel ({y},{x}) channel {c}: expected {expected_val}, got {output_val}"
            )


def test_luminance_threshold(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(
        device, dtype=DType.float32, shape=(100, 100, 1)
    )
    graph = make_graph(
        "luminance_threshold",
        forward=lambda x: ops.luminance_threshold(device, x, threshold=0.5),
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
    assert result.shape == (100, 100, 1)

    input_values = image_tensor.to_numpy()
    output_values = result.to_numpy()
    expected_output = np.where(input_values > 0.5, 1.0, 0.0)
    assert np.allclose(output_values, expected_output, atol=1e-5), (
        "Luminance threshold should be: output = input > threshold ? 1.0 : 0.0"
    )

    test_pixels = [(10, 10), (50, 50), (90, 90)]
    for y, x in test_pixels:
        input_val = input_values[y, x, 0]
        output_val = output_values[y, x, 0]
        expected_val = 1.0 if input_val > 0.5 else 0.0
        assert abs(output_val - expected_val) < 1e-5, (
            f"Pixel ({y},{x}): input={input_val}, expected {expected_val}, got {output_val}"
        )

    unique_values = np.unique(output_values)
    assert len(unique_values) <= 2, "Output should contain only 0s and 1s"
    assert np.all(np.isin(unique_values, [0.0, 1.0])), (
        "Output should contain only 0s and 1s"
    )


def test_rgb_to_luminance(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(
        device, dtype=DType.float32, shape=(100, 100, 3)
    )
    graph = make_graph(
        "rgb_to_luminance",
        forward=lambda x: ops.rgb_to_luminance(device, x),
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
    assert result.shape == (100, 100, 1)

    input_values = image_tensor.to_numpy()
    output_values = result.to_numpy()
    expected_output = (
        input_values[:, :, 0] * 0.2125
        + input_values[:, :, 1] * 0.7154
        + input_values[:, :, 2] * 0.0721
    ).reshape(100, 100, 1)
    assert np.allclose(output_values, expected_output, atol=1e-5), (
        "RGB to luminance should use coefficients: R*0.2125 + G*0.7154 + B*0.0721"
    )

    test_pixels = [(10, 10), (50, 50), (90, 90)]
    for y, x in test_pixels:
        r_val = input_values[y, x, 0]
        g_val = input_values[y, x, 1]
        b_val = input_values[y, x, 2]
        output_val = output_values[y, x, 0]
        expected_val = r_val * 0.2125 + g_val * 0.7154 + b_val * 0.0721
        assert abs(output_val - expected_val) < 1e-5, (
            f"Pixel ({y},{x}): R={r_val}, G={g_val}, B={b_val}, expected {expected_val}, got {output_val}"
        )

    assert output_values.shape[2] == 1, "Output should be single channel"
    assert np.all(output_values >= 0), "Output values should be non-negative"
    assert np.all(output_values <= 1), "Output values should be <= 1"


def test_luminance_to_rgb(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(
        device, dtype=DType.float32, shape=(100, 100, 1)
    )
    graph = make_graph(
        "luminance_to_rgb",
        forward=lambda x: ops.luminance_to_rgb(x),
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

    input_values = image_tensor.to_numpy()
    output_values = result.to_numpy()
    expected_output = np.repeat(input_values, 3, axis=2)
    assert np.allclose(output_values, expected_output, atol=1e-5), (
        "Luminance to RGB should broadcast to 3 identical channels"
    )

    test_pixels = [(10, 10), (50, 50), (90, 90)]
    for y, x in test_pixels:
        input_val = input_values[y, x, 0]
        r_val = output_values[y, x, 0]
        g_val = output_values[y, x, 1]
        b_val = output_values[y, x, 2]

        # All RGB channels should equal the input luminance value
        assert abs(r_val - input_val) < 1e-5, (
            f"Pixel ({y},{x}) R channel: expected {input_val}, got {r_val}"
        )
        assert abs(g_val - input_val) < 1e-5, (
            f"Pixel ({y},{x}) G channel: expected {input_val}, got {g_val}"
        )
        assert abs(b_val - input_val) < 1e-5, (
            f"Pixel ({y},{x}) B channel: expected {input_val}, got {b_val}"
        )

    assert np.allclose(output_values[:, :, 0], output_values[:, :, 1], atol=1e-5), (
        "R and G channels should be identical"
    )
    assert np.allclose(output_values[:, :, 1], output_values[:, :, 2], atol=1e-5), (
        "G and B channels should be identical"
    )
