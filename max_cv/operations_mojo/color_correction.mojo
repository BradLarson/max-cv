import compiler
from std.math import pow
from std.utils.coord import Coord, coord_to_index_list
from extensibility import OutputTensor, InputTensor, foreach
from std.gpu.host import DeviceContext


@compiler.register("brightness")
struct Brightness:
    """Adjusts the brightness of an image."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[...],
        brightness: Float32,
        image: InputTensor[
            dtype=output.dtype, rank=output.rank, static_spec=...
        ],
        ctx: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def add[width: Int](idx: Coord) -> SIMD[image.dtype, width]:
            return image.load[width](idx) + brightness.cast[image.dtype]()

        foreach[add, target=target](output, ctx)


@compiler.register("gamma")
struct Gamma:
    """Adjusts the gamma of an image."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[...],
        gamma: Float32,
        image: InputTensor[
            dtype=output.dtype, rank=output.rank, static_spec=...
        ],
        ctx: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def gamma_kernel[width: Int](idx: Coord) -> SIMD[image.dtype, width]:
            return pow(image.load[width](idx), gamma.cast[image.dtype]())

        foreach[gamma_kernel, target=target](output, ctx)


@compiler.register("luminance")
struct Luminance:
    """Reduce an RGB image to its luminance channel."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[...],
        image: InputTensor[
            dtype=output.dtype, rank=output.rank, static_spec=...
        ],
        ctx: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def luminance[width: Int](idx: Coord) -> SIMD[image.dtype, width]:
            var color_idx = coord_to_index_list(idx)
            color_idx[image.rank - 1] = 0
            var red = image.load[1](color_idx)
            color_idx[image.rank - 1] = 1
            var green = image.load[1](color_idx)
            color_idx[image.rank - 1] = 2
            var blue = image.load[1](color_idx)
            # Values from "Graphics Shaders: Theory and Practice" by Bailey
            # and Cunningham.
            var luminance = red * 0.2125 + green * 0.7154 + blue * 0.0721
            return SIMD[image.dtype, width](luminance)

        foreach[luminance, target=target](output, ctx)
