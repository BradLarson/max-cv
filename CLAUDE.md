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

MAX-CV is an accelerated image processing framework built on MAX and Mojo, inspired by the GPUImage series. It provides a high-level Python API with Mojo-based operations for realtime video processing and machine vision tasks. The framework constructs computational graphs that are compiled to fuse and optimize sequential operations.

## Key Commands

### Development Tasks
- `pixi run test` - Run unit and integration tests
- `pixi run bench` - Run Mojo benchmarks for all operations
- `pixi run mojo ./benchmarks/bench.mojo` - Direct benchmark execution
- `pixi run mojo format file.mojo` - Format Mojo code for correct style

### Examples and Demos
- `pixi run filter-single-image` - Simple image processing demo
- `pixi run showcase` - Show all available operations
- `pixi run showcase [operation] --value [param]` - Run specific operation (e.g., `pixi run showcase pixellate --value 15`)
- `pixi run notebook` - Start Jupyter notebook environment

### Video Processing
- `pixi run showcase_video` - Video processing examples

## Architecture

### Core Components

**Python API Layer (`max_cv/`)**:
- `ImagePipeline` - Main class for constructing image processing graphs
- `io.py` - Image loading/saving utilities with `load_image_into_tensor()`
- `operations/` - Python wrappers for Mojo operations

**Mojo Operations (`operations/`)**:
- Low-level image processing operations implemented in Mojo
- Organized by category: blend, color_correction, draw, edge_detection, effects

**Pipeline Architecture**:
1. Images loaded into tensors on CPU or accelerator
2. `ImagePipeline` context manager constructs computational graph
3. Operations chained together with `pipeline.input_image`
4. Graph compiled once with `pipeline.compile()`
5. Compiled pipeline executed multiple times with `pipeline(tensor)`

### Key Patterns

**Device Management**:
```python
device = CPU() if accelerator_count() == 0 else Accelerator()
```

**Pipeline Construction**:
```python
with ImagePipeline(name, shape, pipeline_dtype, device) as pipeline:
    result = ops.operation(device, pipeline.input_image, params)
    pipeline.output(result)
```

**Tensor Handling**:
- Tensors can reside on CPU or accelerator
- Use `.to(CPU())` to move tensors to host for NumPy conversion
- Internal pipeline dtype typically `DType.float32`

### Operation Categories

- **Color adjustments**: brightness, gamma, luminance_threshold, rgb_to_luminance
- **Image processing**: sobel_edge_detection  
- **Visual effects**: pixellate, gaussian_blur
- **Blending**: add_blend, dissolve_blend, multiply_blend
- **Drawing**: draw_circle

## Dependencies and Environment

- Requires MAX 25.4+ (nightly preferred)
- Uses Pixi for dependency management
- Custom Mojo operations loaded from `operations/` directory
- Testing via pytest, benchmarking via native Mojo

## Recent Updates and Migration Notes

### MAX/Mojo API Migration (June 2025)
The codebase has been updated to work with the latest MAX/Mojo versions. Key changes made:

**Import Changes**:
- `from max.tensor import` → `from tensor_internal import` (for all custom operations)
- Affects: OutputTensor, InputTensor, foreach, ManagedTensorSlice

**Type System Updates**:
- `type=` parameter → `dtype=` in tensor specifications
- `.type` attribute → `.dtype` on tensor objects
- Examples: `InputTensor[type=DType.float32]` → `InputTensor[dtype=DType.float32]`

**Reserved Keywords**:
- `out:` parameter name → `output:` in custom operation execute functions
- The word `out` became reserved in recent Mojo versions

**Files Updated**:
- All operations in `/operations/`: blend.mojo, color_correction.mojo, draw.mojo, edge_detection.mojo, effects.mojo, passthrough.mojo
- All benchmarks in `/benchmarks/`: bench_*.mojo files, common.mojo
- Updated imports and parameter naming consistently across codebase

**Custom Operation Pattern (Modern)**:
```mojo
@compiler.register("operation_name")
struct Operation:
    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32],  # Use 'output', not 'out'
        input: InputTensor[dtype=output.dtype, rank=output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Implementation uses .dtype instead of .type
        pass
```

**Verification Commands**:
- `pixi run test` - All tests pass after migration
- `pixi run mojo benchmarks/bench.mojo` - Benchmarks compile and run successfully
- Both CPU and GPU acceleration confirmed working

These updates ensure compatibility with MAX nightly builds while preserving all existing functionality.