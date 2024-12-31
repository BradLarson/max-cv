from pathlib import Path
from PIL import Image

# Add search path for the max_cv module.
import sys
path_root = Path(__file__).parent.parent
sys.path.append(str(path_root))
print(sys.path)

from max_cv import ImagePipeline, load_image_into_tensor
from max_cv import operations as ops
from max.driver import Accelerator, accelerator_count, CPU
from max.dtype import DType


if __name__ == "__main__":
    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    # Load our initial image into a device Tensor.
    image_path = Path("examples/resources/bucky_birthday_small.jpeg")
    image_tensor = load_image_into_tensor(image_path, device)

    # Configure the image processing pipeline.
    filter_value = 1.5

    with ImagePipeline(
        "filter_single_image",
        image_tensor.shape,
        pipeline_dtype=DType.float32
    ) as pipeline:
        processed_image = ops.gamma(pipeline.input_image, filter_value)
        pipeline.output(processed_image)

    print("Graph:", pipeline._graph)

    # Compile and run the pipeline.
    pipeline.compile(device)
    result = pipeline(image_tensor)

    # Move the results to the host CPU and convert them to NumPy format.
    result = result.to(CPU())
    result_array = result.to_numpy()

    # Save the resulting filtered image.
    im = Image.fromarray(result_array)
    im.save("output.png")

    print("Image pixels:")
    print(result_array)
    print()
