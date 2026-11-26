"""bench 的分布名与 sim.model 的常量名映射。"""
BENCH_TO_MODEL = {
    "uniform": "uniform",
    "sequential": "sequential",
    "hot": "hotkey",
    "zipf": "zipf",
    "mixed": "uniform",   # mixed 主要靠 large_share 体现，分布本身按均匀
}


def to_model(name: str) -> str:
    return BENCH_TO_MODEL.get(name, name)
