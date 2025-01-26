import compiler
from utils.index import IndexList
from max.tensor import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr


@compiler.register("passthrough", num_dps_outputs=1)
struct Passthrough:
    @staticmethod
    fn execute[
        # e.g. "CUDA" or "CPU"
        target: StringLiteral,
    ](
        # as num_dps_outputs=1, the first argument is the "output"
        out: ManagedTensorSlice,
        # starting here are the list of inputs
        image: ManagedTensorSlice[out.type, out.rank],
        # the context is needed for some GPU calls
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[image.rank]) -> SIMD[image.type, width]:
            return image.load[width](idx)

        foreach[func, target=target](out, ctx)

    # You only need to implement this if you do not manually annotate
    # output shapes in the graph.
    @staticmethod
    fn shape(
        x: ManagedTensorSlice,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"
