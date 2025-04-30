import compiler
from utils.index import Index, IndexList
from max.tensor import foreach, InputTensor, OutputTensor
from runtime.asyncrt import DeviceContextPtr
from math import sqrt

@compiler.register("draw_circle")
struct DrawCircle:

    @staticmethod
    fn execute[
        target: StaticString
    ](
        x: OutputTensor,
        image: InputTensor[type=x.type, rank=x.rank],
        radius: Scalar[x.type],
        color: InputTensor[type=x.type, rank=1],
        width: Scalar[x.type],
        center: InputTensor[type=x.type, rank=1],
        ctx: DeviceContextPtr
    ) raises:
        var cx: Scalar[x.type] = center[1]
        var cy: Scalar[x.type] = center[0]
        var inner_dist = radius
        var outer_dist = radius + width

        if color.size() != 3:
            raise "Expected 3 channel color, received: " + String(color.size())

        if center.size() != 2:
            raise "Expected 2 dimensional center point, received: " + String(center.size())

        # TODO: There's definitely a more clever way of doing this
        # once we have the ability to mutate Tensors in place.
        @__copy_capture(cx, cy, inner_dist, outer_dist)
        @parameter
        @always_inline
        fn draw[
            width: Int
        ](idx: IndexList[image.rank]) -> SIMD[image.type, width]:
            var i = (Scalar[x.type](idx[1]) - cx) ** 2
            var j = (Scalar[x.type](idx[0]) - cy) ** 2

            var distance = sqrt(i + j)
            if (outer_dist + 0.5 > distance > inner_dist - 0.5):
                return color[idx[image.rank - 1]]
            return image[idx]

        foreach[draw, target=target, simd_width=1](x, ctx)