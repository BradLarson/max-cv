import compiler
from std.utils.index import Index, IndexList
from tensor import foreach, OutputTensor, InputTensor
from std.runtime.asyncrt import DeviceContextPtr


@compiler.register("flip")
struct Flip:
    """Flips an image horizontally or vertically."""

    @staticmethod
    fn execute[
        target: StaticString,
        flip_code: Int,
    ](
        output: OutputTensor,
        image: InputTensor[dtype = output.dtype, rank = output.rank, ...],
        ctx: DeviceContextPtr,
    ) raises:
        # Have a different kernel for each to avoid branching
        @parameter
        if flip_code >= 0:
            comptime index = 1 if flip_code >= 1 else 0

            @parameter
            @always_inline
            fn single_flip_kernel[
                width: Int
            ](idx: IndexList[image.rank]) -> SIMD[image.dtype, width]:
                var src_idx = idx
                src_idx[index] = image.shape()[index] - src_idx[index] - 1
                return image.load[width](src_idx)

            foreach[single_flip_kernel, target=target](output, ctx)
        else:

            @parameter
            @always_inline
            fn both_flip_kernel[
                width: Int
            ](idx: IndexList[image.rank]) -> SIMD[image.dtype, width]:
                var src_idx = idx
                src_idx[0] = image.shape()[0] - src_idx[0] - 1
                src_idx[1] = image.shape()[1] - src_idx[1] - 1
                return image.load[width](src_idx)

            foreach[both_flip_kernel, target=target](output, ctx)
