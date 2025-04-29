from benchmark import ThroughputMeasure, BenchId, BenchMetric, Bench, Bencher
from operations import Blend
from .common import *
from max.tensor import (
    Input,
    Output,
)

fn run_blend_benchmarks(mut bench: Bench) raises:
    blend(bench)

fn blend(mut bench: Bench) raises:
    var cpu = DeviceContext(api='cpu')
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
                Blend.execute[DT, mode, 'cpu'](
                    outtensor.tensor, 0.5, background_image.tensor, foreground_image.tensor, cpu
                )
            b.iter[run]()

        bench.bench_function[bench_cpu](BenchId('blend_' + mode, 'cpu'), elements)

    bench_blend['add']()
    bench_blend['dissolve']()
    bench_blend['multiply']()