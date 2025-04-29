from benchmark import ThroughputMeasure, BenchId, BenchMetric, Bench, Bencher
from operations import  Luminance, Gamma, Brightness
from .common import *
from max.tensor import (
    Input,
    Output,
)

fn run_color_correction_benchmarks(mut bench: Bench) raises:
    brightness(bench)
    gamma(bench)
    luminance(bench)

fn brightness(mut bench: Bench) raises:
    var cpu = DeviceContext(api='cpu')
    var outtensor = gen_tensor[Output](cpu)
    var intensor = gen_tensor[Input](cpu)
    var els = intensor.size
    var elements = ThroughputMeasure(BenchMetric.elements, els)

    @parameter
    fn bench_cpu(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn run() raises:
            Brightness.execute['cpu'](outtensor.tensor, 0.5, intensor.tensor, cpu)

        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId('brightness', 'cpu'), elements)


fn gamma(mut bench: Bench) raises:
    var cpu = DeviceContext(api='cpu')
    var outtensor = gen_tensor[Output](cpu)
    var intensor = gen_tensor[Input](cpu)
    var els = intensor.size
    var elements = ThroughputMeasure(BenchMetric.elements, els)

    @parameter
    fn bench_cpu(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn run() raises:
            Gamma.execute['cpu'](outtensor.tensor, 0.5, intensor.tensor, cpu)

        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId('gamma', 'cpu'), elements)

fn luminance(mut bench: Bench) raises:
    var cpu = DeviceContext(api='cpu')
    var outtensor = gen_tensor[Output](cpu)
    var intensor = gen_tensor[Input](cpu)
    var els = intensor.size
    var elements = ThroughputMeasure(BenchMetric.elements, els)

    @parameter
    fn bench_cpu(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn run() raises:
            Luminance.execute['cpu'](outtensor.tensor, intensor.tensor, cpu)

        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId('luminance', 'cpu'), elements)