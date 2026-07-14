import compiler
from std.math import sqrt
from std.utils.index import IndexList
from std.utils.coord import Coord, coord_to_index_list
from extensibility import foreach, OutputTensor, InputTensor
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from std.sys import align_of, simd_width_of


# FIXME: This makes a lot of assumptions about the inbound tensor.
def edge_clamped_offset_load[
    width: Int, _rank: Int, type: DType, height_offset: Int, width_offset: Int
](
    tensor: InputTensor[dtype=type, rank=_rank, static_spec=...],
    index: Coord,
) -> SIMD[type, width]:
    var clamped_index = coord_to_index_list(index)
    clamped_index[0] = clamped_index[0] + height_offset
    clamped_index[1] = clamped_index[1] + width_offset

    comptime for i in range(_rank):
        clamped_index[i] = max(0, min(tensor.dim_size(i) - 1, clamped_index[i]))
    return tensor.load[width](clamped_index)


comptime block_dim_size = 16
comptime TPB = block_dim_size * block_dim_size


def sobel_kernel[
    simd_width: Int, dtype: DType, layout: Layout
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    image: LayoutTensor[dtype, layout, MutAnyOrigin],
    strength: Float32,
):
    var x = Int(block_idx.x * block_dim_size + thread_idx.x) * simd_width
    var y = Int(block_idx.y * block_dim_size + thread_idx.y)
    var lx = Int(thread_idx.x) * simd_width
    var ly = Int(thread_idx.y)

    var size_x = Int(image.dim(1))  # Width
    var size_y = Int(image.dim(0))  # Height

    var shared = LayoutTensor[
        dtype,
        # Pad Y by 2, Pad X by 2
        Layout.row_major(block_dim_size + 2, block_dim_size * simd_width + 2),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Pre-calculate clamped Y indices (Scalar)
    var y_c = max(0, min(size_y - 1, y))
    var y_t = max(0, min(size_y - 1, y - 1))
    var y_b = max(0, min(size_y - 1, y + 1))

    # Load center vector
    shared.store[simd_width](ly + 1, lx + 1, image.load[simd_width](y_c, x))

    # Load Top/Bottom Halo vectors (Vector Load)
    if ly == 0:
        shared.store[simd_width](0, lx + 1, image.load[simd_width](y_t, x))
    if ly == block_dim_size - 1:
        shared.store[simd_width](
            block_dim_size + 1, lx + 1, image.load[simd_width](y_b, x)
        )

    # Load Left/Right Halo scalars (Edge threads only)
    if thread_idx.x == 0:
        var x_l = max(0, min(size_x - 1, x - 1))
        # Store to (ly+1, 0)
        shared[ly + 1, 0] = image[y_c, x_l]
        # Corner: Top-Left
        if ly == 0:
            shared[0, 0] = image[y_t, x_l]
        # Corner: Bottom-Left
        if ly == block_dim_size - 1:
            shared[block_dim_size + 1, 0] = image[y_b, x_l]

    if thread_idx.x == block_dim_size - 1:
        var x_r = max(0, min(size_x - 1, x + simd_width))
        # Store to (ly+1, END)
        shared[ly + 1, block_dim_size * simd_width + 1] = image[y_c, x_r]
        # Corner: Top-Right
        if ly == 0:
            shared[0, block_dim_size * simd_width + 1] = image[y_t, x_r]
        # Corner: Bottom-Right
        if ly == block_dim_size - 1:
            shared[block_dim_size + 1, block_dim_size * simd_width + 1] = image[
                y_b, x_r
            ]

    barrier()

    if y < size_y and x < size_x:
        var val_tl = shared.load[simd_width](ly, lx)
        var val_t = shared.load[simd_width](ly, lx + 1)
        var val_tr = shared.load[simd_width](ly, lx + 2)
        var val_l = shared.load[simd_width](ly + 1, lx)
        var val_r = shared.load[simd_width](ly + 1, lx + 2)
        var val_bl = shared.load[simd_width](ly + 2, lx)
        var val_b = shared.load[simd_width](ly + 2, lx + 1)
        var val_br = shared.load[simd_width](ly + 2, lx + 2)

        var gh = -val_tl - 2.0 * val_t - val_tr + val_bl + 2.0 * val_b + val_br
        var gv = -val_bl - 2.0 * val_l - val_tl + val_br + 2.0 * val_r + val_tr

        var mag = sqrt(gh * gh + gv * gv)

        output.store[simd_width](y, x, mag * strength.cast[output.dtype]())


@compiler.register("sobel")
struct SobelEdgeDetection:
    """Performs Sobel edge detection."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[...],
        strength: Float32,
        image: InputTensor[
            dtype=output.dtype, rank=output.rank, static_spec=...
        ],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "gpu":
            var im_shape = image.shape()
            var image_t = image.to_layout_tensor()
            comptime x = image_t.shape[0]()
            comptime y = image_t.shape[1]()
            var image_tensor = image_t.reshape[Layout.row_major(x, y)]()
            var output_tensor = output.to_layout_tensor().reshape[
                Layout.row_major(x, y)
            ]()
            comptime image_layout = image_tensor.layout
            comptime output_layout = output_tensor.layout

            comptime dtype = image_tensor.dtype
            comptime simd_width = simd_width_of[dtype]()

            comptime kernel = sobel_kernel[simd_width, dtype, image_layout]
            ctx.enqueue_function[kernel](
                output_tensor,
                image_tensor,
                strength,
                grid_dim=(
                    im_shape[1] // (block_dim_size * simd_width) + 1,
                    im_shape[0] // block_dim_size + 1,
                    1,
                ),
                block_dim=(block_dim_size, block_dim_size, 1),
            )
        else:

            @parameter
            @always_inline
            def sobel[width: Int](idx: Coord) -> SIMD[image.dtype, width]:
                var top_left = edge_clamped_offset_load[
                    1, height_offset=-1, width_offset=-1
                ](image, idx)
                var top = edge_clamped_offset_load[
                    1, height_offset=-1, width_offset=0
                ](image, idx)
                var top_right = edge_clamped_offset_load[
                    1, height_offset=-1, width_offset=1
                ](image, idx)
                var left = edge_clamped_offset_load[
                    1, height_offset=0, width_offset=-1
                ](image, idx)
                var right = edge_clamped_offset_load[
                    1, height_offset=0, width_offset=1
                ](image, idx)
                var bottom_left = edge_clamped_offset_load[
                    1, height_offset=1, width_offset=-1
                ](image, idx)
                var bottom = edge_clamped_offset_load[
                    1, height_offset=1, width_offset=0
                ](image, idx)
                var bottom_right = edge_clamped_offset_load[
                    1, height_offset=1, width_offset=1
                ](image, idx)
                var h = (
                    -top_left
                    - 2.0 * top
                    - top_right
                    + bottom_left
                    + 2.0 * bottom
                    + bottom_right
                )
                var v = (
                    -bottom_left
                    - 2.0 * left
                    - top_left
                    + bottom_right
                    + 2.0 * right
                    + top_right
                )
                var magnitude = sqrt(h * h + v * v)
                return magnitude * strength.cast[image.dtype]()

            foreach[sobel, target=target, simd_width=1](output, ctx)
