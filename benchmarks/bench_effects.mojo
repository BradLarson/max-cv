from std.benchmark import (
    ThroughputMeasure,
    BenchId,
    BenchMetric,
    Bench,
    Bencher,
)
from operations_mojo import Pixellate
from .common import *
from extensibility import IOSpec


def run_effects_benchmarks(mut bench: Bench) raises:
    pixellate(bench)


def pixellate(mut bench: Bench) raises:
    var cpu = DeviceContext(api="cpu")
    var intensor = gen_tensor[IOSpec.Input](cpu)
    var outtensor = gen_tensor[IOSpec.Output](cpu)

    var els = intensor.size
    var elements = [ThroughputMeasure(BenchMetric.elements, els)]

    @parameter
    def bench_cpu(mut b: Bencher) raises:
        @parameter
        @always_inline
        def run() raises:
            Pixellate.execute[target="cpu"](
                outtensor.tensor, 15, intensor.tensor, cpu
            )

        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId("pixellate", "cpu"), elements)

    comptime if has_accelerator():
        var gpu = DeviceContext()
        var gpu_intensor = gen_tensor[IOSpec.Input](gpu)
        var gpu_outtensor = gen_tensor[IOSpec.Output](gpu)

        @parameter
        def bench_gpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            def run() raises:
                Pixellate.execute[target="gpu"](
                    gpu_outtensor.tensor, 15, gpu_intensor.tensor, gpu
                )
                gpu.synchronize()

            b.iter[run]()

        bench.bench_function[bench_gpu](BenchId("pixellate", "gpu"), elements)
