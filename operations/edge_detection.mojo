import compiler
from builtin.simd import _pow
from math import sqrt
from utils.index import IndexList
from max.tensor import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr


# FIXME: This makes a lot of assumptions about the inbound tensor.
fn edge_clamped_offset_load[
    width: Int, _rank: Int, type: DType, height_offset: Int, width_offset: Int
](tensor: ManagedTensorSlice[type, _rank], index: IndexList[_rank]) -> SIMD[
    type, width
]:
    var clamped_index = index
    clamped_index[0] = clamped_index[0] + height_offset
    clamped_index[1] = clamped_index[1] + width_offset
    for i in range(_rank):
        clamped_index[i] = max(0, min(tensor.dim_size(i) - 1, clamped_index[i]))
    return tensor.load[width](clamped_index)


@compiler.register("sobel", num_dps_outputs=1)
struct SobelEdgeDetection:
    """Performs Sobel edge detection."""

    @staticmethod
    fn execute[
        target: StringLiteral,
    ](
        out: ManagedTensorSlice,
        strength: Float32,
        image: ManagedTensorSlice[out.type, out.rank],
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: IndexList[image.rank]) -> SIMD[image.type, width]:
            var top_left = edge_clamped_offset_load[
                1, height_offset= -1, width_offset= -1
            ](image, idx)
            var top = edge_clamped_offset_load[
                1, height_offset= -1, width_offset=0
            ](image, idx)
            var top_right = edge_clamped_offset_load[
                1, height_offset= -1, width_offset=1
            ](image, idx)
            var left = edge_clamped_offset_load[
                1, height_offset=0, width_offset= -1
            ](image, idx)
            var right = edge_clamped_offset_load[
                1, height_offset=0, width_offset=1
            ](image, idx)
            var bottom_left = edge_clamped_offset_load[
                1, height_offset=1, width_offset= -1
            ](image, idx)
            var bottom = edge_clamped_offset_load[
                1, height_offset=1, width_offset=0
            ](image, idx)
            var bottom_right = edge_clamped_offset_load[
                1, height_offset=1, width_offset=1
            ](image, idx)
            var h = -top_left - 2.0 * top - top_right + bottom_left + 2.0 * bottom + bottom_right
            var v = -bottom_left - 2.0 * left - top_left + bottom_right + 2.0 * right + top_right
            var magnitude = sqrt(h * h + v * v)
            return magnitude * strength.cast[image.type]()

        foreach[func, target=target](out, ctx)
