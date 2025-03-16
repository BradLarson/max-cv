from benchmarks.bench_effects import *
from benchmarks.bench_blend import *
from benchmarks.bench_color_correction import *
from benchmarks.bench_draw import *
from benchmarks.bench_edge_detection import *
from benchmark import Bench

fn main() raises:
    var bench = Bench()
    run_effects_benchmarks(bench)
    run_blend_benchmarks(bench)
    run_color_correction_benchmarks(bench)
    run_draw_benchmarks(bench)
    run_edge_detection_benchmarks(bench)
    bench.dump_report()