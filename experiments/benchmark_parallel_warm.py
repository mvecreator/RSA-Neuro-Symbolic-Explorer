"""Measure one small N with persistent preloaded workers (fixed eight seeds)."""

import concurrent.futures
import json
import multiprocessing
import os
import time

import torch

from benchmark_small_keys import (PARALLEL_P, PARALLEL_Q, PARALLEL_SEEDS,
                                  ROOT, measure_solver)
from TKHD_7_Massive import MDNOracle

ORACLE = None


def initialize_worker():
    global ORACLE
    torch.set_num_threads(1)
    ORACLE = MDNOracle(str(ROOT / "mdn_p_massive.pth"),
                       str(ROOT / "mdn_q_massive.pth"), torch.device("cpu"))


def ping():
    time.sleep(0.25)
    return os.getpid()


def run_seed(seed):
    case = {"bits": 32, "n": PARALLEL_P * PARALLEL_Q}
    return measure_solver(case, "mdn", ORACLE, 500, 1000, seed)


def main():
    context = multiprocessing.get_context("spawn")
    runs = []
    for workers in (1, 2, 4, 8):
        cold_start = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=initialize_worker
        ) as pool:
            warmed = set()
            for _ in range(3):
                warmed.update(f.result() for f in
                              [pool.submit(ping) for _ in range(workers)])
                if len(warmed) == workers:
                    break
            setup_s = time.perf_counter() - cold_start
            start = time.perf_counter()
            futures = [pool.submit(run_seed, seed) for seed in PARALLEL_SEEDS]
            completions = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                completions.append({
                    "seed": result["seed"], "success": result["success"],
                    "completed_s": time.perf_counter() - start,
                    "solver_s": result["seconds"],
                    "candidate_checks": result["candidate_checks"],
                })
            all_done_s = time.perf_counter() - start
        run = {
            "workers": workers, "warmed_processes": len(warmed),
            "setup_s": setup_s, "all_done_s": all_done_s,
            "first_success_s": min((x["completed_s"] for x in completions
                                    if x["success"]), default=None),
            "success_count": sum(x["success"] for x in completions),
            "total_candidate_checks": sum(x["candidate_checks"]
                                          for x in completions),
            "completed": completions,
        }
        runs.append(run)
        print(f"workers={workers}, warmed={len(warmed)}, "
              f"success={run['success_count']}/8, "
              f"first={run['first_success_s']:.3f}s, all={all_done_s:.3f}s",
              flush=True)
    (ROOT / "results" / "parallel_one_n_warm.json").write_text(
        json.dumps({"case": {"n": PARALLEL_P * PARALLEL_Q,
                             "p": PARALLEL_P, "q": PARALLEL_Q},
                    "seeds": list(PARALLEL_SEEDS), "runs": runs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
