import compiler
from std.utils.coord import Coord, coord_to_index_list
from extensibility import foreach, OutputTensor, InputTensor
from std.gpu.host import DeviceContext


@compiler.register("pixellate")
struct Pixellate:
    """Pixellates an image into small squares."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[...],
        pixel_width: Int32,
        image: InputTensor[
            dtype=output.dtype, rank=output.rank, static_spec=...
        ],
        ctx: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def pixellate[width: Int](idx: Coord) -> SIMD[image.dtype, width]:
            var pixel_idx = coord_to_index_list(idx)
            pixel_idx[0] = (pixel_idx[0] // Int(pixel_width)) * Int(pixel_width)
            pixel_idx[1] = (pixel_idx[1] // Int(pixel_width)) * Int(pixel_width)
            return image.load[width](pixel_idx)

        foreach[pixellate, target=target](output, ctx)
