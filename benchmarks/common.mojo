from std.math import iota
from std.max.driver import cpu
from std.gpu.host import DeviceContext, DeviceBuffer
from extensibility import (
    ManagedTensorSlice,
    StaticTensorSpec,
    get_row_major_tensor_spec_static,
    IO,
    IOSpec,
)
from layout import IntTuple, to_index_list
from layout.int_tuple import product
from std.memory import AddressSpace
from std.memory import UnsafePointer
from std.random import rand
from std.utils import IndexList
from std.sys import size_of, has_accelerator, CompilationTarget

comptime dtype = DType.float32
comptime rank = 3
comptime tspec = get_row_major_tensor_spec_static[dtype, 3, 3840, 2160, 3]()

comptime tspec_lum = get_row_major_tensor_spec_static[dtype, 3, 3840, 2160, 1]()

comptime point_spec = get_row_major_tensor_spec_static[dtype, 1, 2]()
comptime color_spec = get_row_major_tensor_spec_static[dtype, 1, 3]()


def gen_tensor[
    mut: Bool, input: IO, //, iospec: IOSpec[mut, input]
](ctx: DeviceContext) raises -> BenchTensor[iospec, tspec]:
    return BenchTensor[iospec, tspec](ctx).rand()


def gen_color_tensor(
    ctx: DeviceContext,
) raises -> BenchTensor[IOSpec.Input, color_spec]:
    return BenchTensor[IOSpec.Input, color_spec](ctx).rand()


@fieldwise_init
struct BenchTensor[
    dtype: DType,
    rank: Int,
    mut: Bool,
    input: IO,
    //,
    io_spec: IOSpec[mut, input],
    static_spec: StaticTensorSpec[dtype, rank, _],
](Copyable, Movable):
    comptime tensor_type = ManagedTensorSlice[
        io_spec=Self.io_spec, static_spec=Self.static_spec
    ]
    comptime buffer_type = DeviceBuffer[Self.dtype]
    comptime ptr_type = UnsafePointer[Scalar[Self.dtype]]
    comptime size = product(Self.static_spec.shape_tuple)

    var tensor: Self.tensor_type
    var buffer: Self.buffer_type

    def __init__(out self, ctx: DeviceContext) raises:
        self.buffer = ctx.enqueue_create_buffer[Self.dtype](Self.size)
        ctx.synchronize()

        self.tensor = ManagedTensorSlice[
            io_spec=Self.io_spec, static_spec=Self.static_spec
        ](
            self.buffer.unsafe_ptr(),
            to_index_list[Self.rank](Self.static_spec.shape_tuple),
            to_index_list[Self.rank](Self.static_spec.strides_tuple),
        )

    def unsafe_ptr(self) -> Self.ptr_type:
        return self.buffer.unsafe_ptr()

    def rand(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            rand(host_buffer.unsafe_ptr(), Self.size)
            return self.copy()

    def iota(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            iota(host_buffer.unsafe_ptr(), Self.size)
            return self.copy()
