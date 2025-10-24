from benchmark import ThroughputMeasure, BenchId, BenchMetric, Bench, Bencher
from operations_mojo import DrawCircle
from .common import *
from tensor import (
    Input,
    Output,
)


fn run_draw_benchmarks(mut bench: Bench) raises:
    draw_circle(bench)


fn draw_circle(mut bench: Bench) raises:
    var cpu = DeviceContext(api="cpu")
    var intensor = gen_tensor[Input](cpu)
    var outtensor = gen_tensor[Output](cpu)
    var color = gen_color_tensor(cpu)
    var center = BenchTensor[Input, point_spec](cpu)
    center.tensor[0] = intensor.size // 2
    center.tensor[1] = intensor.size // 2

    var els = intensor.size
    var elements = ThroughputMeasure(BenchMetric.elements, els)

    @parameter
    fn bench_cpu(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn run() raises:
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

    @parameter
    if False:  # GPU crashes - complex indexing with center tensor access
        var gpu = DeviceContext()
        var gpu_intensor = gen_tensor[Input](gpu)
        var gpu_outtensor = gen_tensor[Output](gpu)
        var gpu_color = gen_color_tensor(gpu)
        var gpu_center = BenchTensor[Input, point_spec](gpu)
        gpu_center.tensor[0] = gpu_intensor.size // 2
        gpu_center.tensor[1] = gpu_intensor.size // 2

        @parameter
        fn bench_gpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            fn run() raises:
                DrawCircle.execute["gpu"](
                    gpu_outtensor.tensor,
                    gpu_intensor.tensor,
                    120.0,
                    gpu_color.tensor,
                    5.0,
                    gpu_center.tensor,
                    gpu,
                )

            b.iter[run]()

        bench.bench_function[bench_gpu](BenchId("draw_circle", "gpu"), elements)
