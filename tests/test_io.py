from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from max_cv import load_image_into_tensor, normalize_image, restore_image
from pathlib import Path
from .common import generate_test_tensor, run_graph

def test_load_image_into_tensor() -> None:
    device = CPU()
    image_path = Path("examples/resources/bucky_birthday_small.jpeg")
    image_tensor = load_image_into_tensor(image_path, device)
    image_tensor = image_tensor.to(CPU())

    shape = image_tensor.shape
    assert shape == (600, 450, 3)

def test_normalize_image(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.uint8)
    graph = Graph(
        "normalize",
        forward=lambda x: normalize_image(x, DType.float32),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == (4, 6, 3)

def test_restore_image(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.float32)
    graph = Graph(
        "restore",
        forward=lambda x: restore_image(x),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.uint8
    assert result.shape == (4, 6, 3)
