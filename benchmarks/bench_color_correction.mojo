from std.benchmark import (
    ThroughputMeasure,
    BenchId,
    BenchMetric,
    Bench,
    Bencher,
)
from operations_mojo import Luminance, Gamma, Brightness
from .common import *
from extensibility import IOSpec


def run_color_correction_benchmarks(mut bench: Bench) raises:
    brightness(bench)
    gamma(bench)
    luminance(bench)


def brightness(mut bench: Bench) raises:
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
            Brightness.execute["cpu"](
                outtensor.tensor, 0.5, intensor.tensor, cpu
            )

        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId("brightness", "cpu"), elements)

    comptime if has_accelerator():
        var gpu = DeviceContext()
        var gpu_outtensor = gen_tensor[IOSpec.Output](gpu)
        var gpu_intensor = gen_tensor[IOSpec.Input](gpu)

        @parameter
        def bench_gpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            def run() raises:
                Brightness.execute["gpu"](
                    gpu_outtensor.tensor, 0.5, gpu_intensor.tensor, gpu
                )
                gpu.synchronize()

            b.iter[run]()

        bench.bench_function[bench_gpu](BenchId("brightness", "gpu"), elements)


def gamma(mut bench: Bench) raises:
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
            Gamma.execute["cpu"](outtensor.tensor, 0.5, intensor.tensor, cpu)

        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId("gamma", "cpu"), elements)

    comptime if has_accelerator():
        var gpu = DeviceContext()
        var gpu_outtensor = gen_tensor[IOSpec.Output](gpu)
        var gpu_intensor = gen_tensor[IOSpec.Input](gpu)

        @parameter
        def bench_gpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            def run() raises:
                Gamma.execute["gpu"](
                    gpu_outtensor.tensor, 0.5, gpu_intensor.tensor, gpu
                )
                gpu.synchronize()

            b.iter[run]()

        bench.bench_function[bench_gpu](BenchId("gamma", "gpu"), elements)


def luminance(mut bench: Bench) raises:
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
            Luminance.execute["cpu"](outtensor.tensor, intensor.tensor, cpu)

        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId("luminance", "cpu"), elements)

    comptime if has_accelerator():
        var gpu = DeviceContext()
        var gpu_outtensor = gen_tensor[IOSpec.Output](gpu)
        var gpu_intensor = gen_tensor[IOSpec.Input](gpu)

        @parameter
        def bench_gpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            def run() raises:
                Luminance.execute["gpu"](
                    gpu_outtensor.tensor, gpu_intensor.tensor, gpu
                )
                gpu.synchronize()

            b.iter[run]()

        bench.bench_function[bench_gpu](BenchId("luminance", "gpu"), elements)
