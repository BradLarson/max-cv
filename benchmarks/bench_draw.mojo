from std.benchmark import (
    ThroughputMeasure,
    BenchId,
    BenchMetric,
    Bench,
    Bencher,
)
from operations_mojo import DrawCircle
from .common import *
from extensibility import IOSpec


def run_draw_benchmarks(mut bench: Bench) raises:
    draw_circle(bench)


def draw_circle(mut bench: Bench) raises:
    var cpu = DeviceContext(api="cpu")
    var intensor = gen_tensor[IOSpec.Input](cpu)
    var outtensor = gen_tensor[IOSpec.Output](cpu)
    var color = gen_color_tensor(cpu)
    var center = BenchTensor[IOSpec.Input, point_spec](cpu)
    center.tensor[0] = Float32(intensor.size // 2)
    center.tensor[1] = Float32(intensor.size // 2)

    var els = intensor.size
    var elements = [ThroughputMeasure(BenchMetric.elements, els)]

    @parameter
    def bench_cpu(mut b: Bencher) raises:
        @parameter
        @always_inline
        def run() raises:
            DrawCircle.execute["cpu"](
                outtensor.tensor,
                intensor.tensor,
                120.0,
                color.tensor,
                5.0,
                center.tensor,
                cpu,
            )

        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId("draw_circle", "cpu"), elements)

    comptime if has_accelerator():
        var gpu = DeviceContext()
        var gpu_intensor = gen_tensor[IOSpec.Input](gpu)
        var gpu_outtensor = gen_tensor[IOSpec.Output](gpu)

        @parameter
        def bench_gpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            def run() raises:
                DrawCircle.execute["gpu"](
                    gpu_outtensor.tensor,
                    gpu_intensor.tensor,
                    120.0,
                    color.tensor,
                    5.0,
                    center.tensor,
                    gpu,
                )
                gpu.synchronize()

            b.iter[run]()

        bench.bench_function[bench_gpu](
            BenchId("draw_circle", "gpu"), List(elements)
        )
