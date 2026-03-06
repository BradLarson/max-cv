import compiler
from std.utils.index import Index, IndexList
from tensor import foreach, InputTensor, OutputTensor
from std.runtime.asyncrt import DeviceContextPtr
from std.math import sqrt


@compiler.register("draw_circle")
struct DrawCircle:
    @staticmethod
    fn execute[
        target: StaticString
    ](
        output: OutputTensor,
        image: InputTensor[dtype = output.dtype, rank = output.rank, ...],
        radius: Scalar[output.dtype],
        color: InputTensor[dtype = output.dtype, rank=1, ...],
        width: Scalar[output.dtype],
        center: InputTensor[dtype = output.dtype, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        if color.size() != 3:
            raise Error(
                "Expected 3 channel color, received: " + String(color.size())
            )

        if center.size() != 2:
            raise Error(
                "Expected 2 dimensional center point, received: "
                + String(center.size())
            )

        var cx = center.load[1](IndexList[1](1))
        var cy = center.load[1](IndexList[1](0))
        var r_color = color.load[1](IndexList[1](0))
        var g_color = color.load[1](IndexList[1](1))
        var b_color = color.load[1](IndexList[1](2))
        var inner_dist = radius
        var outer_dist = radius + width

        @__copy_capture(
            cx, cy, inner_dist, outer_dist, r_color, g_color, b_color
        )
        @parameter
        @always_inline
        fn draw[
            simd_width: Int
        ](idx: IndexList[image.rank]) -> SIMD[image.dtype, simd_width]:
            var i = (Scalar[output.dtype](idx[1]) - cx) ** 2
            var j = (Scalar[output.dtype](idx[0]) - cy) ** 2

            var distance = sqrt(i + j)
            var in_ring = (distance < outer_dist + 0.5) and (
                distance > inner_dist - 0.5
            )
            if in_ring:
                var channel = idx[image.rank - 1]
                if channel == 0:
                    return r_color
                elif channel == 1:
                    return g_color
                else:
                    return b_color
            return image.load[simd_width](idx)

        foreach[draw, target=target, simd_width=1](output, ctx)
