from benchmark import ThroughputMeasure, BenchId, BenchMetric, Bench, Bencher
from operations import DrawCircle
from .common import *
from max.tensor import (
    Input,
    Output,
)

fn run_draw_benchmarks(mut bench: Bench) raises:
    draw_circle(bench)

fn draw_circle(mut bench: Bench) raises:
    var cpu = DeviceContext(api='cpu')
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
            DrawCircle.execute['cpu'](
                outtensor.tensor,
                intensor.tensor,
                120.0,
                color.tensor,
                5.0,
                center.tensor,
                cpu
            )
        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId('draw_circle', 'cpu'), elements)