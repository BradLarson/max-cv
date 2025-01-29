import compiler
from utils.index import IndexList
from max.tensor import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr


fn _add[
    width: Int, float_dtype: DType
](
    foreground_pixel: SIMD[float_dtype, width],
    background_pixel: SIMD[float_dtype, width],
    intensity: SIMD[float_dtype, 1],
) -> SIMD[float_dtype, width]:
    return foreground_pixel + background_pixel


fn _mix[
    width: Int, float_dtype: DType
](
    foreground_pixel: SIMD[float_dtype, width],
    background_pixel: SIMD[float_dtype, width],
    intensity: SIMD[float_dtype, 1],
) -> SIMD[float_dtype, width]:
    return foreground_pixel * intensity + background_pixel * (1.0 - intensity)


fn _multiply[
    width: Int, float_dtype: DType
](
    foreground_pixel: SIMD[float_dtype, width],
    background_pixel: SIMD[float_dtype, width],
    intensity: SIMD[float_dtype, 1],
) -> SIMD[float_dtype, width]:
    return foreground_pixel * background_pixel


@compiler.register("blend", num_dps_outputs=1)
struct Blend:
    """Performs a two-image blend, using a specified blend function."""

    @staticmethod
    fn execute[
        type: DType,
        blend_mode: StringLiteral,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice[type],
        intensity: Float32,
        background_image: ManagedTensorSlice[type, out.rank],
        foreground_image: ManagedTensorSlice[type, out.rank],
        ctx: MojoCallContextPtr,
    ):
        var converted_intensity = intensity.cast[foreground_image.type]()

        @parameter
        @always_inline
        fn blend[
            width: Int
        ](idx: IndexList[foreground_image.rank]) -> SIMD[
            foreground_image.type, width
        ]:
            var foreground_pixel = foreground_image.load[width](idx)
            var background_pixel = background_image.load[width](idx)

            @parameter
            if blend_mode == "add":
                return _add[float_dtype = foreground_image.type](
                    foreground_pixel, background_pixel, converted_intensity
                )
            elif blend_mode == "dissolve":
                return _mix[float_dtype = foreground_image.type](
                    foreground_pixel, background_pixel, converted_intensity
                )
            elif blend_mode == "multiply":
                return _multiply[float_dtype = foreground_image.type](
                    foreground_pixel, background_pixel, converted_intensity
                )
            else:
                # TODO: Better error handling here.
                print("Unsupported blend mode:", blend_mode)
                return foreground_pixel

        foreach[blend, target=target](out, ctx)

        _ = converted_intensity
