from benchmark import ThroughputMeasure, BenchId, BenchMetric, Bench, Bencher
from operations import SobelEdgeDetection
from .common import *
from tensor_internal import (
    Input,
    Output,
)


fn run_edge_detection_benchmarks(mut bench: Bench) raises:
    sobel(bench)


fn sobel(mut bench: Bench) raises:
    var cpu = DeviceContext(api="cpu")
    var intensor = gen_tensor[Input](cpu)
    var outtensor = gen_tensor[Output](cpu)

    var els = intensor.size
    var elements = ThroughputMeasure(BenchMetric.elements, els)

    @parameter
    fn bench_cpu(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn run() raises:
            SobelEdgeDetection.execute["cpu"](
                outtensor.tensor, 0.5, intensor.tensor, cpu
            )

        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId("sobel", "cpu"), elements)

    @parameter
    if has_accelerator():
        var gpu = DeviceContext()
        var gpu_intensor = gen_tensor[Input](gpu)
        var gpu_outtensor = gen_tensor[Output](gpu)

        @parameter
        fn bench_gpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            fn run() raises:
                SobelEdgeDetection.execute["gpu"](
                    gpu_outtensor.tensor, 0.5, gpu_intensor.tensor, gpu
                )

            b.iter[run]()

        bench.bench_function[bench_gpu](BenchId("sobel", "gpu"), elements)
