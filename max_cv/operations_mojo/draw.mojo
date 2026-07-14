import compiler
from std.utils.index import IndexList
from std.utils.coord import Coord, coord_to_index_list
from extensibility import foreach, InputTensor, OutputTensor
from std.gpu.host import DeviceContext
from std.math import sqrt


@compiler.register("draw_circle")
struct DrawCircle:
    @staticmethod
    def execute[
        target: StaticString
    ](
        output: OutputTensor[...],
        image: InputTensor[
            dtype=output.dtype, rank=output.rank, static_spec=...
        ],
        radius: Scalar[output.dtype],
        color: InputTensor[dtype=output.dtype, rank=1, static_spec=...],
        width: Scalar[output.dtype],
        center: InputTensor[dtype=output.dtype, rank=1, static_spec=...],
        ctx: DeviceContext,
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
        def draw[simd_width: Int](idx: Coord) -> SIMD[image.dtype, simd_width]:
            var pixel_idx = coord_to_index_list(idx)
            var i = (Scalar[output.dtype](pixel_idx[1]) - cx) ** 2
            var j = (Scalar[output.dtype](pixel_idx[0]) - cy) ** 2

            var distance = sqrt(i + j)
            var in_ring = (distance < outer_dist + 0.5) and (
                distance > inner_dist - 0.5
            )
            if in_ring:
                var channel = pixel_idx[image.rank - 1]
                if channel == 0:
                    return r_color
                elif channel == 1:
                    return g_color
                else:
                    return b_color
            return image.load[simd_width](idx)

        foreach[draw, target=target, simd_width=1](output, ctx)
