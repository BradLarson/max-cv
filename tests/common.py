import numpy as np
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.graph import Graph
from max.engine import InferenceSession

def generate_test_tensor(device: Device, dtype: DType, shape: tuple = (4, 6, 3)) -> Tensor:
    # TODO: Actual values.
    image_array = np.zeros(shape, dtype=dtype.to_numpy())
    return Tensor.from_numpy(image_array).to(device) 

def run_graph(graph: Graph, input: Tensor, session: InferenceSession) -> Tensor:
    model = session.load(graph)
    result = model.execute(input)[0]
    return result.to(CPU())