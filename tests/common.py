import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType

def generate_test_tensor(device: Device, dtype: DType) -> Tensor:
    # TODO: Actual values.
    image_array = np.zeros((4, 6, 3), dtype=dtype.to_numpy())
    return Tensor.from_numpy(image_array).to(device) 
