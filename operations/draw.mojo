import compiler
from utils.index import Index, IndexList
from tensor_internal import foreach, InputTensor, OutputTensor
from runtime.asyncrt import DeviceContextPtr
from math import sqrt


@compiler.register("draw_circle")
struct DrawCircle:
    @staticmethod
    fn execute[
        target: StaticString
    ](
        output: OutputTensor,
        image: InputTensor[dtype = output.dtype, rank = output.rank],
        radius: Scalar[output.dtype],
        color: InputTensor[dtype = output.dtype, rank=1],
        width: Scalar[output.dtype],
        center: InputTensor[dtype = output.dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        var cx: Scalar[output.dtype] = center[1]
        var cy: Scalar[output.dtype] = center[0]
        var inner_dist = radius
        var outer_dist = radius + width

        if color.size() != 3:
            raise "Expected 3 channel color, received: " + String(color.size())

        if center.size() != 2:
            raise "Expected 2 dimensional center point, received: " + String(
                center.size()
            )

        # TODO: There's definitely a more clever way of doing this
        # once we have the ability to mutate Tensors in place.
        @__copy_capture(cx, cy, inner_dist, outer_dist)
        @parameter
        @always_inline
        fn draw[
            width: Int
        ](idx: IndexList[image.rank]) -> SIMD[image.dtype, width]:
            var i = (Scalar[output.dtype](idx[1]) - cx) ** 2
            var j = (Scalar[output.dtype](idx[0]) - cy) ** 2

            var distance = sqrt(i + j)
            if outer_dist + 0.5 > distance > inner_dist - 0.5:
                return color[idx[image.rank - 1]]
            return image[idx]

        foreach[draw, target=target, simd_width=1](output, ctx)
