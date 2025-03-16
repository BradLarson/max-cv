from benchmarks.bench_effects import *
from benchmark import Bench

fn main() raises:
    var bench = Bench()
    run_effects_benchmarks(bench)
    bench.dump_report()