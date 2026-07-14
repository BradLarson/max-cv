from std.benchmark import (
    ThroughputMeasure,
    BenchId,
    BenchMetric,
    Bench,
    Bencher,
)
from operations_mojo import Flip
from .common import *
from extensibility import IOSpec


comptime FLIP_VERTICAL = 0
comptime FLIP_HORIZONTAL = 1
comptime FLIP_BOTH = -1


def run_transform_benchmarks(mut bench: Bench) raises:
    bench_flip[FLIP_VERTICAL, "flip_vertical"](bench)
    bench_flip[FLIP_HORIZONTAL, "flip_horizontal"](bench)
    bench_flip[FLIP_BOTH, "flip_both"](bench)


def bench_flip[flip_code: Int, name: StringLiteral](mut bench: Bench) raises:
    var cpu = DeviceContext(api="cpu")
    var outtensor = gen_tensor[IOSpec.Output](cpu)
    var intensor = gen_tensor[IOSpec.Input](cpu)
    var els = intensor.size
    var elements = [ThroughputMeasure(BenchMetric.elements, els)]

    @parameter
    def bench_cpu(mut b: Bencher) raises:
        @parameter
        @always_inline
        def run() raises:
            Flip.execute["cpu", flip_code](
                outtensor.tensor, intensor.tensor, cpu
            )

        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId(name, "cpu"), elements)

    comptime if has_accelerator():
        var gpu = DeviceContext()
        var gpu_outtensor = gen_tensor[IOSpec.Output](gpu)
        var gpu_intensor = gen_tensor[IOSpec.Input](gpu)

        @parameter
        def bench_gpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            def run() raises:
                Flip.execute["gpu", flip_code](
                    gpu_outtensor.tensor, gpu_intensor.tensor, gpu
                )
                gpu.synchronize()

            b.iter[run]()

        bench.bench_function[bench_gpu](BenchId(name, "gpu"), elements)
