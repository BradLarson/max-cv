from std.benchmark import (
    ThroughputMeasure,
    BenchId,
    BenchMetric,
    Bench,
    Bencher,
)
from operations_mojo import SobelEdgeDetection
from .common import *
from extensibility import IOSpec


def run_edge_detection_benchmarks(mut bench: Bench) raises:
    sobel(bench)


def sobel(mut bench: Bench) raises:
    var cpu = DeviceContext(api="cpu")
    var intensor = BenchTensor[IOSpec.Input, tspec_lum](cpu).rand()
    var outtensor = BenchTensor[IOSpec.Output, tspec_lum](cpu).rand()

    var els = intensor.size
    var elements = [ThroughputMeasure(BenchMetric.elements, els)]

    @parameter
    def bench_cpu(mut b: Bencher) raises:
        @parameter
        @always_inline
        def run() raises:
            SobelEdgeDetection.execute["cpu"](
                outtensor.tensor, 0.5, intensor.tensor, cpu
            )

        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId("sobel", "cpu"), elements)

    comptime if has_accelerator():
        var gpu = DeviceContext()
        var gpu_intensor = BenchTensor[IOSpec.Input, tspec_lum](gpu).rand()
        var gpu_outtensor = BenchTensor[IOSpec.Output, tspec_lum](gpu).rand()

        @parameter
        def bench_gpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            def run() raises:
                SobelEdgeDetection.execute["gpu"](
                    gpu_outtensor.tensor, 0.5, gpu_intensor.tensor, gpu
                )
                gpu.synchronize()

            b.iter[run]()

        bench.bench_function[bench_gpu](BenchId("sobel", "gpu"), elements)
