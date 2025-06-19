from benchmark import ThroughputMeasure, BenchId, BenchMetric, Bench, Bencher
from operations import Blend
from .common import *
from tensor_internal import (
    Input,
    Output,
)


fn run_blend_benchmarks(mut bench: Bench) raises:
    blend(bench)


fn blend(mut bench: Bench) raises:
    var cpu = DeviceContext(api="cpu")
    var background_image = gen_tensor[Input](cpu)
    var foreground_image = gen_tensor[Input](cpu)
    var outtensor = gen_tensor[Output](cpu)

    var els = background_image.size
    var elements = ThroughputMeasure(BenchMetric.elements, els)

    alias DT = DType.float32

    @parameter
    fn bench_blend[mode: StringLiteral]() raises:
        @parameter
        @always_inline
        fn bench_cpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            fn run() raises:
                Blend.execute[DT, mode, "cpu"](
                    outtensor.tensor,
                    0.5,
                    background_image.tensor,
                    foreground_image.tensor,
                    cpu,
                )

            b.iter[run]()

        bench.bench_function[bench_cpu](
            BenchId("blend_" + mode, "cpu"), elements
        )

        @parameter
        if (
            has_accelerator() and mode != "dissolve"
        ):  # dissolve has GPU memory issues
            var gpu = DeviceContext()
            var gpu_background_image = gen_tensor[Input](gpu)
            var gpu_foreground_image = gen_tensor[Input](gpu)
            var gpu_outtensor = gen_tensor[Output](gpu)

            @parameter
            @always_inline
            fn bench_gpu(mut b: Bencher) raises:
                @parameter
                @always_inline
                fn run() raises:
                    Blend.execute[DT, mode, "gpu"](
                        gpu_outtensor.tensor,
                        0.5,
                        gpu_background_image.tensor,
                        gpu_foreground_image.tensor,
                        gpu,
                    )

                b.iter[run]()

            bench.bench_function[bench_gpu](
                BenchId("blend_" + mode, "gpu"), elements
            )

    bench_blend["add"]()
    bench_blend["dissolve"]()
    bench_blend["multiply"]()
