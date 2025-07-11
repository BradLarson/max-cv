# MAX-CV #

Brad Larson

[@bradlarson.bsky.social](https://bsky.app/profile/bradlarson.bsky.social)

[@bradlarson@hachyderm.io](https://hachyderm.io/@bradlarson)

## Overview ##

MAX-CV is an exploration of accelerated image processing using
[MAX](https://docs.modular.com/max/) and
[Mojo](https://docs.modular.com/mojo/manual/). Inspiration for this framework
has been drawn from the [GPUImage](https://github.com/BradLarson/GPUImage)
series of Objective-C and Swift frameworks based on OpenGL or Metal.

The objective of the framework is to make it as easy as possible to set up and
perform realtime video processing or machine vision against image or video
sources, running operations on a range of CPUs, GPUs, or more. Operations are
defined in a platform-independent manner using the Mojo language, with a
high-level Python API to ease integration into existing code. Graphs of image
processing operations are constructed and compiled to fuse and optimize
sequential operations.

## License ##

Apache License v2.0 with LLVM Exceptions.

## Technical requirements ##

[MAX](https://docs.modular.com/max/) 25.4 or newer (nightly preferred). The
command-line tool `pixi` will handle all dependency setup for MAX, and can be
installed via

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

- OS: Ubuntu 22.04 or newer, macOS 14.0 or newer
- NVIDIA GPU: Turing or newer architectures
- AMD GPU: RDNA3+ or CDNA3 architectures

## How to use ##

The `max_cv` library builds on [MAX](https://docs.modular.com/max/), and the
easiest way to install and work with MAX is through the `pixi` tool listed
above. This repository contains a `pixi` project defined in the
`pixi.toml`, and that can be built upon to add new Python applications
using `max_cv`.

Within the application, import the appropriate components from MAX and
`max_cv`:

```python
from pathlib import Path
from max_cv import ImagePipeline, load_image_into_tensor
from max_cv import operations as ops
from max.driver import Accelerator, accelerator_count, CPU
from max.dtype import DType
```

Images can be loaded from standard file formats directly into tensors that
reside on CPU or on an accelerator:

```python
# Place the graph on a GPU, if available. Fall back to CPU if not.
device = CPU() if accelerator_count() == 0 else Accelerator()

# Load our initial image into a device Tensor.
image_path = Path("examples/resources/bucky_birthday_small.jpeg")
image_tensor = load_image_into_tensor(image_path, device)
```

An image processing pipeline is constructed using a context manager, with an
input image chained through operations. The final result is marked using
`output()`. The internal datatype used for intermediate steps in the image
pipeline can be set using `pipeline_dtype`. The following sets up an image
processing pipeline that adjusts the brightness of an incoming image:

```python
with ImagePipeline(
    "adjust_brightness",
    image_tensor.shape,
    pipeline_dtype=DType.float32,
    device=device
) as pipeline:
    processed_image = ops.brightness(device, pipeline.input_image, 0.5)
    pipeline.output(processed_image)
```

The image processing pipeline is defined as a computational graph that must be
first compiled by MAX. This compilation only needs to be performed once for a
given pipeline, and will be cached between subsequent runs. Graph compilation
lets subsequent operations be fused together, as well as applying other
optimizations.

```python
pipeline.compile()
```

Once the pipeline has been compiled, it can be run quickly against as many
input images as desired:

```python
result = pipeline(image_tensor)
```

The result is a tensor that resides on the accelerator (if one was used) for
efficient use in other MAX pipelines, such as in AI models. To access the
image and save it to disk, you can move the tensor from device to the host,
convert to a NumPy array, and save that using standard libraries:

```python
from PIL import Image

result = result.to(CPU())
result_array = result.to_numpy()
im = Image.fromarray(result_array)
im.save("output.png")
```

## Notebook ##

A Jupyter notebook environment is provided that can be used for
experimentation with image processing and MAX. To activate it, run:

```sh
pixi run notebook
```

A [sample notebook](examples/filter_single_image.ipynb) has been provided to
showcase how to use the library.

## Examples ##

Two examples are provided, starting with a simple demonstration of how to run
an image through a processing pipeline and obtain a resulting image. Assuming
that you have installed the Magic command-line tool referenced above, building
and running this example is as simple as calling the following in the current
directory:

```sh
pixi run filter-single-image
```

A more advanced example acts as a showcase of all of the image operations in
the library, with each operation as a subcommand of the main `showcase.py`
example. To see all available image operations, run

```sh
pixi run showcase
```

and as an example of how to execute a single operation and show the resulting
image to the display:

```sh
pixi run showcase pixellate --value 15
```

## Tests ##

The set of unit and integration tests for the library can be run using the
following invocation:

```sh
pixi run test
```

## Built-in operations ##

Operations are currently being added, with only a small set available at
present. Here are the ones that are currently functional:

### Color adjustments ###

- **brightness**: Adjusts the brightness of the image.
  - *brightness*: The adjusted brightness (-1.0 - 1.0, with 0.0 as the default)

- **gamma**: Adjusts the gamma of an image.
  - *gamma*: The gamma adjustment to apply (0.0 - 3.0, with 1.0 as the default)

- **luminance_threshold**: Sets a pixel to black if below a luminance
  threshold, white otherwise.
  - *threshold*: The threshold to test against, ranging from 0.0 - 1.0.

- **rgb_to_luminance**: Reduces an image to its luminance (grayscale). The
  result is a single-channel image, usable directly by anything that expects a
  luminance-only input. For operations expecting an RGB input, you'll need to
  use `luminance_to_rgb()` to convert the image back to a three-channel one.

### Image processing ###

- **sobel_edge_detection**: Sobel edge detection, with edges highlighted in white.
  - *strength*: Adjusts the dynamic range of the filter. Higher values lead to
  stronger edges, but can saturate the intensity colorspace. Default is 1.0.

### Visual effects ###

- **pixellate**: Applies a pixellation effect on an image.
  - *pixel_width*: How large the square pixels will be in the final image.
- **gaussian_blur**: Applies a guassian blur filter to the image.
  - *kernel_size*: The size of the blur kernel.
  - *sigma*: The standard deviation used for computing the kernel.
  - *padding*: Whether to pad the input image to avoid losing pixels at the edge if the image.

### Blending ###

- **add_blend**: Applies an additive blend of two images.

- **dissolve_blend**: Applies a dissolve blend of two images.
  - *intensity*: The degree with which the foreground image overrides the
  background (0.0 - 1.0, with 0.5 as the default).

- **multiply_blend**: Applies a multiply blend of two images.

### Drawing ###

- **draw_circle**: Draws a circle on the image.
  - *radius*: The radius of the circle.
  - *color*: The color of the circle as an rgb value.
  - *width*: The thickness of the circle.
  - *center*: The position of the center of the circle. Defaults to the center of the image.
