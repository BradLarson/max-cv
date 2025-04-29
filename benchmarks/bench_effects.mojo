from benchmark import ThroughputMeasure, BenchId, BenchMetric, Bench, Bencher
from operations import Pixellate
from .common import *
from max.tensor import (
    Input,
    Output,
)

fn run_effects_benchmarks(mut bench: Bench) raises:
    pixellate(bench)

fn pixellate(mut bench: Bench) raises:
    var cpu = DeviceContext(api='cpu')
    var intensor = gen_tensor[Input](cpu)
    var outtensor = gen_tensor[Output](cpu)

    var els = intensor.size
    var elements = ThroughputMeasure(BenchMetric.elements, els)

    @parameter
    fn bench_cpu(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn run() raises:
            Pixellate.execute[target='cpu'](
                outtensor.tensor, 15, intensor.tensor, cpu
            )
        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId('pixellate', 'cpu'), elements)
