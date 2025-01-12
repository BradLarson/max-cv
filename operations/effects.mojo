import compiler
from builtin.simd import _pow
from math import sqrt
from utils.index import Index, IndexList
from tensor_utils import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr


@compiler.register("pixellate", num_dps_outputs=1)
struct Pixellate:
    """Pixellates an image into small squares."""

    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice,
        pixel_width: Int32,
        image: ManagedTensorSlice[out.type, out.rank],
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: IndexList[image.rank]) -> SIMD[image.type, width]:
            var pixel_idx = idx
            pixel_idx[0] = (pixel_idx[0] // Int(pixel_width)) * Int(pixel_width)
            pixel_idx[1] = (pixel_idx[1] // Int(pixel_width)) * Int(pixel_width)
            return image.load[width](pixel_idx)

        foreach[func, synchronous, target](out, ctx)
