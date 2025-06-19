from benchmarks.bench_effects import *
from benchmarks.bench_blend import *
from benchmarks.bench_color_correction import *
from benchmarks.bench_draw import *
from benchmarks.bench_edge_detection import *
from benchmark import Bench


fn main() raises:
    from benchmarks.common import has_accelerator
    from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator
    from gpu.host import DeviceContext

    print("MAX-CV Benchmarks")
    print("================")

    @parameter
    if has_accelerator():
        print("Using GPU:", DeviceContext().name())
    else:
        print("Only benchmarking on CPU.")
    print()

    var bench = Bench()
    run_effects_benchmarks(bench)
    run_blend_benchmarks(bench)
    run_color_correction_benchmarks(bench)
    run_draw_benchmarks(bench)
    run_edge_detection_benchmarks(bench)
    bench.dump_report()
