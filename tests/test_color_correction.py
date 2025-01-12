from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
import max_cv.operations as ops
from .common import generate_test_tensor, run_graph

def test_brightness(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.float32)
    graph = Graph(
        "brightness",
        forward=lambda x: ops.brightness(x, 0.5),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == (4, 6, 3)

def test_gamma(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.float32)
    graph = Graph(
        "gamma",
        forward=lambda x: ops.gamma(x, 1.5),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == (4, 6, 3)

def test_rgb_to_luminance(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.float32)
    graph = Graph(
        "rgb_to_luminance",
        forward=lambda x: ops.rgb_to_luminance(x),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == (4, 6, 1)

def test_luminance_to_rgb(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.float32, channels=1)
    graph = Graph(
        "luminance_to_rgb",
        forward=lambda x: ops.luminance_to_rgb(x),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == (4, 6, 3)
