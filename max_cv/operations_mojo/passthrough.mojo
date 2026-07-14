import compiler
from std.utils.coord import Coord
from extensibility import (
    ManagedTensorSlice,
    foreach,
    OutputTensor,
    InputTensor,
)
from std.gpu.host import DeviceContext


@compiler.register("passthrough")
struct Passthrough:
    @staticmethod
    def execute[
        # e.g. "CUDA" or "CPU"
        target: StaticString,
    ](
        # as num_dps_outputs=1, the first argument is the "output"
        output: OutputTensor[...],
        # starting here are the list of inputs
        image: InputTensor[
            dtype=output.dtype, rank=output.rank, static_spec=...
        ],
        # the context is needed for some GPU calls
        ctx: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[image.dtype, width]:
            return image.load[width](idx)

        foreach[func, target=target](output, ctx)
