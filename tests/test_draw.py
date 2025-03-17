from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
import max_cv.operations as ops
from .common import generate_test_tensor, run_graph

def test_draw(session: InferenceSession) -> None:
    device = CPU()
    input_shape = (50, 30, 3)
    image_tensor = generate_test_tensor(
        device,
        dtype=DType.float32,
        shape=input_shape,
    )
    graph = Graph(
        "draw_circle",
        forward=lambda x: ops.draw_circle(x, 4, (255, 0, 0), 1),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape),
        ],
    )

    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == input_shape
    assert result != image_tensor