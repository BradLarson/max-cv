import compiler
from utils.index import IndexList
from tensor import (
    ManagedTensorSlice,
    foreach,
    OutputTensor,
    InputTensor,
)
from runtime.asyncrt import DeviceContextPtr


@compiler.register("passthrough")
struct Passthrough:
    @staticmethod
    fn execute[
        # e.g. "CUDA" or "CPU"
        target: StaticString,
    ](
        # as num_dps_outputs=1, the first argument is the "output"
        output: OutputTensor,
        # starting here are the list of inputs
        image: InputTensor[dtype = output.dtype, rank = output.rank],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: IndexList[image.rank]) -> SIMD[image.dtype, width]:
            return image.load[width](idx)

        foreach[func, target=target](output, ctx)

    # You only need to implement this if you do not manually annotate
    # output shapes in the graph.
    @staticmethod
    fn shape(
        x: InputTensor,
    ) raises -> IndexList[x.rank]:
        raise Error("NotImplemented")
