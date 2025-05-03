from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, DeviceRef
import max_cv.operations as ops
from .common import generate_test_tensor, run_graph, make_graph

def test_brightness(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.float32)
    graph = make_graph(
        "brightness",
        forward=lambda x: ops.brightness(device, x, 0.5),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape, device=DeviceRef.from_device(device)),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == (4, 6, 3)

def test_gamma(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.float32)
    graph = make_graph(
        "gamma",
        forward=lambda x: ops.gamma(device, x, 1.5),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape, device=DeviceRef.from_device(device)),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == (4, 6, 3)

def test_luminance_threshold(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.float32, shape=(4, 6, 1))
    graph = make_graph(
        "luminance_threshold",
        forward=lambda x: ops.luminance_threshold(device, x, threshold=0.5),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape, device=DeviceRef.from_device(device)),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == (4, 6, 1)

def test_rgb_to_luminance(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.float32)
    graph = make_graph(
        "rgb_to_luminance",
        forward=lambda x: ops.rgb_to_luminance(device, x),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape, device=DeviceRef.from_device(device)),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == (4, 6, 1)

def test_luminance_to_rgb(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.float32, shape=(4, 6, 1))
    graph = make_graph(
        "luminance_to_rgb",
        forward=lambda x: ops.luminance_to_rgb(x),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape, device=DeviceRef.from_device(device)),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == (4, 6, 3)
