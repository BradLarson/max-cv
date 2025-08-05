import compiler
from builtin.simd import _pow
from utils.index import IndexList
from tensor_internal import OutputTensor, InputTensor, foreach
from runtime.asyncrt import DeviceContextPtr


@compiler.register("brightness")
struct Brightness:
    """Adjusts the brightness of an image."""

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        brightness: Float32,
        image: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn add[
            width: Int
        ](idx: IndexList[image.rank]) -> SIMD[image.dtype, width]:
            return image.load[width](idx) + brightness.cast[image.dtype]()

        foreach[add, target=target](output, ctx)


@compiler.register("gamma")
struct Gamma:
    """Adjusts the gamma of an image."""

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        gamma: Float32,
        image: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn pow[
            width: Int
        ](idx: IndexList[image.rank]) -> SIMD[image.dtype, width]:
            return _pow(image.load[width](idx), gamma)

        foreach[pow, target=target](output, ctx)


@compiler.register("luminance")
struct Luminance:
    """Reduce an RGB image to its luminance channel."""

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        image: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn luminance[
            width: Int
        ](idx: IndexList[image.rank]) -> SIMD[image.dtype, width]:
            var color_idx = idx
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
