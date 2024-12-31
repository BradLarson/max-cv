import compiler
from builtin.simd import _pow
from utils.index import IndexList
from tensor_utils import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr

@compiler.register("brightness", num_dps_outputs=1)
struct Brightness:
    """Adjusts the brightness of an image."""
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice,
        brightness: Float32,
        image: ManagedTensorSlice[out.type, out.rank],
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[image.rank]) -> SIMD[image.type, width]:
            return image.load[width](idx) + brightness.cast[image.type]()

        foreach[func, synchronous, target](out, ctx)

@compiler.register("gamma", num_dps_outputs=1)
struct Gamma:
    """Adjusts the gamma of an image."""
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice,
        gamma: Float32,
        image: ManagedTensorSlice[out.type, out.rank],
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[image.rank]) -> SIMD[image.type, width]:
            return _pow(image.load[width](idx), gamma)

        foreach[func, synchronous, target](out, ctx)
