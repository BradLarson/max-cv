from pathlib import Path
from PIL import Image

# Add search path for the max_cv module.
import sys

path_root = Path(__file__).parent.parent
sys.path.append(str(path_root))

from max_cv import ImagePipeline, load_image_into_tensor  # noqa: E402
from max_cv import operations as ops  # noqa: E402
from max.driver import Accelerator, accelerator_count, CPU  # noqa: E402
from max.dtype import DType  # noqa: E402


if __name__ == "__main__":
    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    # Load our initial image into a device Tensor.
    image_path = Path("examples/resources/bucky_birthday_small.jpeg")
    image_tensor = load_image_into_tensor(image_path, device)

    # Configure the image processing pipeline.
    filter_value = 0.5

    with ImagePipeline(
        "filter_single_image",
        image_tensor.shape,
        pipeline_dtype=DType.float32,
        device=device,
    ) as pipeline:
        processed_image = ops.pixellate(device, pipeline.input_image, pixel_width=20)
        pipeline.output(processed_image)

    # Compile and run the pipeline.
    print("Compiling graph...")
    pipeline.compile()
    print("Compilation finished. Running image pipeline...")
    result = pipeline(image_tensor)
    print("Processing finished.")

    # Move the results to the host CPU and convert them to NumPy format.
    result = result.to(CPU())
    result_array = result.to_numpy()

    # Save the resulting filtered image.
    im = Image.fromarray(result_array)
    im.save("output.png")

    print("Image pixels:")
    print(result_array)
    print()
