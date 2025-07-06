import click
from collections.abc import Callable
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image

# Add search path for the max_cv module.
import sys

path_root = Path(__file__).parent.parent
sys.path.append(str(path_root))

from max_cv import ImagePipeline, load_image_into_tensor  # noqa: E402
from max_cv import operations as ops  # noqa: E402
from max.driver import Accelerator, CPU, Device  # noqa: E402
from max.dtype import DType  # noqa: E402
from max.graph import TensorValue  # noqa: E402


@click.group()
def showcase():
    pass


# Color adjustment operations.


@showcase.command(name="brightness")
@click.option(
    "--value",
    type=float,
    default=0,
    show_default=True,
    help="Brightness adjustment, from -1.0 - 1.0.",
)
def brightness(value):
    print("Adjusting brightness by:", value)
    run_pipeline(operations=lambda device, input: ops.brightness(device, input, value))


@showcase.command(name="gamma")
@click.option(
    "--value",
    type=float,
    default=1.0,
    show_default=True,
    help="Gamma adjustment.",
)
def gamma(value):
    print("Adjusting gamma by:", value)
    run_pipeline(operations=lambda device, input: ops.gamma(device, input, value))


@showcase.command(name="luminance_threshold")
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Luminance threshold, from 0.0 - 1.0.",
)
def luminance_threshold(threshold):
    print("Thresholding luminance at:", threshold)

    def thresholding(device: Device, input: TensorValue) -> TensorValue:
        processed_image = ops.rgb_to_luminance(device, input)
        return ops.luminance_threshold(device, processed_image, threshold=threshold)

    run_pipeline(operations=thresholding)


@showcase.command(name="rgb_to_luminance")
def rgb_to_luminance():
    print("Reducing image to luminance channel.")
    run_pipeline(operations=lambda device, input: ops.rgb_to_luminance(device, input))


# Edge detection.


@showcase.command(name="sobel")
@click.option(
    "--value",
    type=float,
    default=1.0,
    show_default=True,
    help="Edge strength.",
)
def sobel(value):
    print("Performing Sobel edge detection with strength:", value)

    def edge_detection(device: Device, input: TensorValue) -> TensorValue:
        processed_image = ops.rgb_to_luminance(device, input)
        return ops.sobel_edge_detection(device, processed_image, strength=1.0)

    run_pipeline(operations=edge_detection)


# Effects.


@showcase.command(name="pixellate")
@click.option(
    "--value",
    type=int,
    default=15,
    show_default=True,
    help="Pixel width.",
)
def pixellate(value):
    print("Pixellating image with pixel width:", value)
    run_pipeline(operations=lambda device, input: ops.pixellate(device, input, value))


@showcase.command(name="gaussian_blur")
@click.option(
    "--kernel_size",
    type=int,
    default=16,
    show_default=True,
    help="the size of the convolution kernel",
)
@click.option(
    "--sigma", type=float, default=4.0, show_default=True, help="gaussian filter stddev"
)
def guassian(kernel_size, sigma):
    print("Running gaussian blur with size and sigma:", kernel_size, sigma)
    run_pipeline(
        operations=lambda device, input: ops.gaussian_blur(
            device, input, kernel_size, sigma
        )
    )


# Blends.


@showcase.command(name="add_blend")
def add_blend():
    print("Applying additive blend on two images.")
    run_pipeline(
        operations=lambda device, inputs: ops.add_blend(device, inputs[0], inputs[1]),
        num_inputs=2,
    )


@showcase.command(name="dissolve_blend")
@click.option(
    "--intensity",
    type=float,
    default=0.5,
    show_default=True,
    help="Blend strength.",
)
def dissolve_blend(intensity):
    print("Applying dissolve blend on two images with intensity:", intensity)
    run_pipeline(
        operations=lambda device, inputs: ops.dissolve_blend(
            device, inputs[0], inputs[1], intensity
        ),
        num_inputs=2,
    )


@showcase.command(name="multiply_blend")
def multiply_blend():
    print("Applying multiply blend on two images.")
    run_pipeline(
        operations=lambda device, inputs: ops.multiply_blend(
            device, inputs[0], inputs[1]
        ),
        num_inputs=2,
    )


# Drawing.


@showcase.command(name="draw_circle")
@click.option(
    "--radius", type=int, default=120, show_default=True, help="radius of the circle"
)
@click.option(
    "--width", type=int, default=5, show_default=True, help="thickness of the circle"
)
@click.option(
    "--center",
    type=int,
    nargs=2,
    default=None,
    show_default=True,
    help="center point of the circle",
)
@click.option(
    "--color",
    type=int,
    nargs=3,
    default=(255, 0, 0),
    show_default=True,
    help="RGB color value of the circle",
)
def draw_circle(radius, color, width, center):
    print(
        f"drawing a circle on the image with radius {radius}, color {color}, width: {width}, center point {center or 'default'}"
    )
    run_pipeline(
        operations=lambda device, input: ops.draw_circle(
            device, input, radius, color, width, center
        )
    )


def run_pipeline(operations: Callable, num_inputs: int = 1):
    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device: Device

    try:
        # Unsupported accelerators will throw an error here
        device = Accelerator()
    except ValueError:
        device = CPU()

    # Load our initial image into a device Tensor.
    image_path = Path("examples/resources/bucky_birthday_small.jpeg")
    image_tensor = load_image_into_tensor(image_path, device)

    if num_inputs > 1:
        # Load a foreground blend image for the cases that need a second image.
        bg_image_path = Path("examples/resources/wisconsin_institutes_small.jpeg")
        bg_image_tensor = load_image_into_tensor(bg_image_path, device)

    # Configure the image processing pipeline.
    with ImagePipeline(
        "filter_single_image",
        image_tensor.shape,
        pipeline_dtype=DType.float32,
        device=device,
        num_inputs=num_inputs,
    ) as pipeline:
        if num_inputs == 1:
            processed_image = operations(device, pipeline.input_image)
        else:
            processed_image = operations(device, pipeline.input_images)
        pipeline.output(processed_image)

    # Compile and run the pipeline.
    print("Compiling graph...")
    pipeline.compile()
    print("Compilation finished. Running image pipeline...")
    if num_inputs == 1:
        result = pipeline(image_tensor)
    else:
        result = pipeline(image_tensor, bg_image_tensor)
    print("Processing finished.")

    # Move the results to the host CPU and convert them to NumPy format.
    result = result.to(CPU())
    result_array = result.to_numpy()

    # Save the resulting filtered image.
    im = Image.fromarray(result_array)
    im.save("output.png")

    plt.imshow(result_array, interpolation="nearest")
    plt.show()


if __name__ == "__main__":
    showcase()
