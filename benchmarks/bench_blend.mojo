from std.benchmark import (
    ThroughputMeasure,
    BenchId,
    BenchMetric,
    Bench,
    Bencher,
)
from operations_mojo import Blend
from .common import *
from extensibility import IOSpec


def run_blend_benchmarks(mut bench: Bench) raises:
    blend(bench)


def blend(mut bench: Bench) raises:
    var cpu = DeviceContext(api="cpu")
    var background_image = gen_tensor[IOSpec.Input](cpu)
    var foreground_image = gen_tensor[IOSpec.Input](cpu)
    var outtensor = gen_tensor[IOSpec.Output](cpu)

    var els = background_image.size
    var elements = [ThroughputMeasure(BenchMetric.elements, els)]

    comptime DT = DType.float32

    @parameter
    def bench_blend[mode: StringLiteral]() raises:
        @parameter
        @always_inline
        def bench_cpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            def run() raises:
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

        comptime if has_accelerator():
            var gpu = DeviceContext()
            var gpu_background_image = gen_tensor[IOSpec.Input](gpu)
            var gpu_foreground_image = gen_tensor[IOSpec.Input](gpu)
            var gpu_outtensor = gen_tensor[IOSpec.Output](gpu)

            @parameter
            @always_inline
            def bench_gpu(mut b: Bencher) raises:
                @parameter
                @always_inline
                def run() raises:
                    Blend.execute[DT, mode, "gpu"](
                        gpu_outtensor.tensor,
                        0.5,
                        gpu_background_image.tensor,
                        gpu_foreground_image.tensor,
                        gpu,
                    )
                    gpu.synchronize()

                b.iter[run]()

            bench.bench_function[bench_gpu](
                BenchId("blend_" + mode, "gpu"), elements
            )

    bench_blend["add"]()
    bench_blend["dissolve"]()
    bench_blend["multiply"]()
