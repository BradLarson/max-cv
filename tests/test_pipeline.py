from max_cv import ImagePipeline
from max_cv import operations as ops
from max.driver import CPU
from max.dtype import DType
from .common import generate_test_tensor

def test_no_ops_pipeline() -> None:
    # TODO: Run on all available devices.
    device = CPU()

    image_tensor = generate_test_tensor(device, dtype=DType.uint8)

    with ImagePipeline(
        "passthrough",
        image_tensor.shape,
        pipeline_dtype=DType.float32,
        device=device
    ) as pipeline:
        pipeline.output(pipeline.input_image)

    pipeline.compile()
    result = pipeline(image_tensor)
    result = result.to(CPU())

    shape = result.shape
    assert shape == (4, 6, 3)

    result_array = result.to_numpy()
    assert result_array[0, 0, 0] == 0
