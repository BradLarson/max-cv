import compiler
from std.utils.coord import Coord
from extensibility import OutputTensor, InputTensor, foreach
from std.gpu.host import DeviceContext


def _add[
    width: Int, float_dtype: DType
](
    foreground_pixel: SIMD[float_dtype, width],
    background_pixel: SIMD[float_dtype, width],
    intensity: SIMD[float_dtype, 1],
) -> SIMD[float_dtype, width]:
    return foreground_pixel + background_pixel


def _mix[
    width: Int, float_dtype: DType
](
    foreground_pixel: SIMD[float_dtype, width],
    background_pixel: SIMD[float_dtype, width],
    intensity: SIMD[float_dtype, 1],
) -> SIMD[float_dtype, width]:
    return foreground_pixel * intensity + background_pixel * (1.0 - intensity)


def _multiply[
    width: Int, float_dtype: DType
](
    foreground_pixel: SIMD[float_dtype, width],
    background_pixel: SIMD[float_dtype, width],
    intensity: SIMD[float_dtype, 1],
) -> SIMD[float_dtype, width]:
    return foreground_pixel * background_pixel


@compiler.register("blend")
struct Blend:
    """Performs a two-image blend, using a specified blend function."""

    @staticmethod
    def execute[
        type: DType,
        blend_mode: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=type, static_spec=...],
        intensity: Float32,
        background_image: InputTensor[
            dtype=type, rank=output.rank, static_spec=...
        ],
        foreground_image: InputTensor[
            dtype=type, rank=output.rank, static_spec=...
        ],
        ctx: DeviceContext,
    ) raises:
        var converted_intensity = intensity.cast[foreground_image.dtype]()

        @__copy_capture(converted_intensity)
        @parameter
        @always_inline
        def blend[
            width: Int
        ](idx: Coord) -> SIMD[foreground_image.dtype, width]:
            var foreground_pixel = foreground_image.load[width](idx)
            var background_pixel = background_image.load[width](idx)

            comptime if blend_mode == "add":
                return _add[float_dtype=foreground_image.dtype](
                    foreground_pixel, background_pixel, converted_intensity
                )
            elif blend_mode == "dissolve":
                return _mix[float_dtype=foreground_image.dtype](
                    foreground_pixel, background_pixel, converted_intensity
                )
            elif blend_mode == "multiply":
                return _multiply[float_dtype=foreground_image.dtype](
                    foreground_pixel, background_pixel, converted_intensity
                )
            else:
                # TODO: Better error handling here.
                print("Unsupported blend mode:", blend_mode)
                return foreground_pixel

        foreach[blend, target=target](output, ctx)
