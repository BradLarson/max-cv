from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, DeviceRef
import max_cv.operations as ops
from .common import generate_test_tensor, run_graph, make_graph

def test_pixellate(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.float32)
    graph = make_graph(
        "pixellate",
        forward=lambda x: ops.pixellate(device, x, 10),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape, device=DeviceRef.from_device(device)),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == (4, 6, 3)

def test_gaussian_blur(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.float32)

    graph = make_graph(
        "gaussian blur",
        forward=lambda x: ops.gaussian_blur(device, x, 3, 3.0, True),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape, device=DeviceRef.from_device(device)),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    # This check is a bit iffy depending on the input shape, could be off by 1
    # in some cases
    assert result.shape == (4, 6, 3)
    assert image_tensor != result