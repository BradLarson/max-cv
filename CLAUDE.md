# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Primary References

When working on this project, make sure to refer to the latest documentation:

- Comprehensive Modular docs: <https://docs.modular.com/llms.txt>
- Mojo docs: <https://docs.modular.com/llms-mojo.txt>
- MAX Python API docs: <https://docs.modular.com/llms-python.txt>

The `modular` GitHub repository has the latest MAX and Mojo code, so either
work from a local checkout, especially the examples in `modular/examples`, or
look at <https://github.com/modular/modular/>

## Project Overview

MAX-CV is an accelerated image processing framework built on MAX and Mojo, inspired by the GPUImage series. It provides a high-level Python API with Mojo-based custom operations for realtime video processing and machine vision tasks. The framework constructs computational graphs that are compiled to fuse and optimize sequential operations, targeting both CPU and GPU acceleration.

## Key Commands

### Development Tasks
- `pixi run test` - Run unit and integration tests via pytest
- `pixi run bench` - Run Mojo benchmarks for all operations
- `pixi run format` - Format both Python (ruff) and Mojo code
- `pixi run format_mojo` - Format only Mojo code (80 char line width)

### Examples and Demos
- `pixi run filter-single-image` - Simple image processing demo
- `pixi run showcase [operation]` - Run specific operation
  - Examples: `pixi run showcase pixellate --value 15`
  - `pixi run showcase brightness --value 0.3`
  - `pixi run showcase gaussian_blur --kernel_size 16 --sigma 4.0`
- `pixi run -e notebook notebook` - Start Jupyter notebook environment

### Video Processing
- `pixi run showcase_video` - Video processing examples (requires OpenCV)

## Architecture

### Core Components

**Python API Layer (`max_cv/`)**:
- `ImagePipeline` - Main class for constructing image processing graphs
- `io.py` - Image loading/saving utilities with `load_image_into_tensor()`
- `operations/` - Python wrappers that invoke Mojo custom operations via `ops.custom()`

**Mojo Custom Operations (`max_cv/operations_mojo/`)**:
- Low-level image processing operations implemented in Mojo
- Uses `@compiler.register("op_name")` decorator to register custom ops
- Operations use `foreach` with element-wise kernels for GPU/CPU execution
- Organized by category: blend, color_correction, draw, edge_detection, effects, transform

**Pipeline Architecture**:
1. Images loaded into `Buffer` objects on CPU or accelerator
2. `ImagePipeline` context manager constructs a MAX `Graph`
3. Operations chained together with `pipeline.input_image` (normalized to 0.0-1.0 float32)
4. Graph compiled once with `pipeline.compile()` to create an optimized `Model`
5. Compiled pipeline executed multiple times with `pipeline(buffer)`
6. Output restored to uint8 0-255 colorspace automatically

### Key Patterns

**Device Management**:
```python
from max.driver import Accelerator, CPU, accelerator_count

device = CPU() if accelerator_count() == 0 else Accelerator()
```

**Pipeline Construction**:
```python
from max_cv import ImagePipeline, load_image_into_tensor, operations as ops
from max.dtype import DType

image_tensor = load_image_into_tensor(image_path, device)

with ImagePipeline(
    "pipeline_name",
    image_tensor.shape,
    pipeline_dtype=DType.float32,
    device=device,
) as pipeline:
    result = ops.brightness(device, pipeline.input_image, 0.5)
    pipeline.output(result)

pipeline.compile()
result = pipeline(image_tensor)
```

**Buffer Handling**:
- Use `Buffer.from_numpy()` to create buffers from NumPy arrays
- Use `buffer.to(device)` to move buffers between CPU and accelerator
- Use `buffer.to(CPU()).to_numpy()` to convert back for display/saving

**Custom Operation Pattern (Mojo)**:
```mojo
@compiler.register("operation_name")
struct OperationName:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        param: Float32,
        image: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def kernel[width: Int](idx: Coord) -> SIMD[image.dtype, width]:
            # Process pixels here
            return image.load[width](idx) + param

        foreach[kernel, target=target](output, ctx)
```

Imports for the above: `from extensibility import InputTensor, OutputTensor,
foreach`, `from std.gpu.host import DeviceContext`, and
`from std.utils.coord import Coord`.

Kernels receive a `Coord`. To do index arithmetic (offsets, clamping, per-channel
loads), convert it with `coord_to_index_list(idx)` from `std.utils.coord`, mutate
the resulting `IndexList`, and pass that to `load`.

### Operation Categories

- **Color adjustments**: brightness, gamma, luminance_threshold, rgb_to_luminance, luminance_to_rgb
- **Edge detection**: sobel_edge_detection
- **Visual effects**: pixellate, gaussian_blur
- **Blending**: add_blend, dissolve_blend, multiply_blend (two-image operations)
- **Drawing**: draw_circle
- **Transforms**: flip (vertical, horizontal, or both)

### Multi-Input Pipelines

For operations requiring multiple images (like blends):
```python
with ImagePipeline(..., num_inputs=2) as pipeline:
    result = ops.dissolve_blend(device, pipeline.input_images[0], pipeline.input_images[1], 0.5)
    pipeline.output(result)

result = pipeline(image1_buffer, image2_buffer)
```

## Dependencies and Environment

- Requires MAX 26.2+ nightly (`https://conda.modular.com/max-nightly` channel)
- Uses Pixi for dependency and environment management
- Platforms: macOS ARM64, Linux ARM64, Linux x86_64
- Custom Mojo operations loaded from `max_cv/operations_mojo/` directory
- Testing via pytest, benchmarking via native Mojo in `benchmarks/`
