from max_cv import ImagePipeline
from max.driver import CPU
from max.dtype import DType
import numpy as np
from .common import generate_test_tensor


def test_no_ops_pipeline() -> None:
    # TODO: Run on all available devices.
    device = CPU()

    image_tensor = generate_test_tensor(device, dtype=DType.uint8)
    input_array = image_tensor.to_numpy()

    with ImagePipeline(
        "passthrough", image_tensor.shape, pipeline_dtype=DType.float32, device=device
    ) as pipeline:
        pipeline.output(pipeline.input_image)

    pipeline.compile()
    result = pipeline(image_tensor)
    result = result.to(CPU())

    shape = result.shape
    assert shape == (4, 6, 3)

    result_array = result.to_numpy()

    # Pipeline should pass through data unchanged (with potential dtype conversion)
    # Convert back to uint8 for comparison since pipeline normalizes to float32 internally

    if result_array.dtype != input_array.dtype:
        # If result is float32 in 0-1 range, convert back to uint8
        if result_array.dtype == np.float32 and np.max(result_array) <= 1.0:
            result_array_uint8 = (result_array * 255).astype(np.uint8)
        else:
            result_array_uint8 = result_array.astype(np.uint8)
    else:
        result_array_uint8 = result_array

    # Verify the data passed through correctly
    assert np.allclose(result_array_uint8, input_array, atol=1), (
        "Pipeline should pass through data unchanged"
    )
