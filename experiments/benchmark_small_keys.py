"""Reproducible small-key audit of the imported TKHD+MDN prototype.

Run from the repository root with a CPU PyTorch environment:
    python experiments/benchmark_small_keys.py --samples-per-size 20

No large RSA key is attacked here. The benchmark checks small semiprimes only.
"""

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import sympy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TKHD_7_Massive import MDNOracle  # noqa: E402
from solver_api import solve_n  # noqa: E402

SEED = 20260929
PARALLEL_P = 40009
PARALLEL_Q = 60013
PARALLEL_SEEDS = tuple(range(8))


class NoJumpOracle:
    """Ablation: keep the current coordinates at each scheduled oracle call."""

    def get_top_candidates(self, n, x, y, top_k=5, num_attempts=1000):
        return [[float(x[i].item())] for i in range(n.shape[0])], [
            [float(y[i].item())] for i in range(n.shape[0])
        ], [[0.0] for _ in range(n.shape[0])]


class UniformHyperbolaOracle:
    """Random control with the same N and bit-range information as the model."""

    def get_top_candidates(self, n, x, y, top_k=5, num_attempts=1000):
        all_p, all_q, all_energies = [], [], []
        for value in n.cpu().numpy().flatten():
            n_int = int(value)
            prime_bits = n_int.bit_length() // 2
            low = max(1 << (prime_bits - 1), math.ceil(n_int / (1 << prime_bits)))
            high = math.isqrt(n_int)
            p = np.random.uniform(low, high, size=num_attempts)
            q = value / p
            energy = 100.0 * (np.sin(np.pi * p) ** 2 + np.sin(np.pi * q) ** 2)
            indices = np.argsort(energy)[:top_k]
            all_p.append(p[indices].tolist())
            all_q.append(q[indices].tolist())
            all_energies.append(energy[indices].tolist())
        return all_p, all_q, all_energies


def archived_moduli():
    tensor = torch.load(ROOT / "rsa_massive_test.pt", map_location="cpu", weights_only=True)
    return {int(row[2].item()) for row in tensor}


def generate_cases(bits, count, excluded, seed=SEED):
    rng = random.Random(seed + bits)
    prime_bits = bits // 2
    low, high = 1 << (prime_bits - 1), 1 << prime_bits
    cases = []
    seen = set(excluded)
    while len(cases) < count:
        p = int(sympy.nextprime(rng.randrange(low, high - 1)))
        q = int(sympy.nextprime(rng.randrange(low, high - 1)))
        if p >= high or q >= high or p == q:
            continue
        p, q = sorted((p, q))
        n = p * q
        if n.bit_length() != bits or n in seen:
            continue
        seen.add(n)
        cases.append({"bits": bits, "n": n, "p": p, "q": q})
    return cases


def trial_division(n):
    checks = 1
    if n % 2 == 0:
        return 2, n // 2, checks
    limit = math.isqrt(n)
    for d in range(3, limit + 1, 2):
        checks += 1
        if n % d == 0:
            return d, n // d, checks
    return None, None, checks


def measure_solver(case, mode, oracle, max_steps, num_attempts, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    start = time.perf_counter()
    result = solve_n(case["n"], 16, oracle=oracle, max_steps=max_steps,
                     num_attempts=num_attempts)
    elapsed = time.perf_counter() - start
    p, q = result.get("p"), result.get("q")
    return {
        "bits": case["bits"], "n": case["n"], "mode": mode, "seed": seed,
        "success": bool(p and q and p * q == case["n"] and p > 1 and q > 1),
        "p": p, "q": q, "seconds": elapsed,
        "candidate_checks": result.get("candidate_checks"),
        "oracle_calls": result.get("oracle_calls"),
        "steps": len(result.get("steps", [])),
        "error": result.get("error"),
    }


def measure_trial(case):
    start = time.perf_counter()
    p, q, checks = trial_division(case["n"])
    elapsed = time.perf_counter() - start
    return {
        "bits": case["bits"], "n": case["n"], "mode": "trial_division",
        "success": bool(p and q and p * q == case["n"]),
        "p": p, "q": q, "seconds": elapsed,
        "candidate_checks": checks, "oracle_calls": 0, "steps": None,
    }


def parallel_worker(seed, max_steps, num_attempts):
    torch.set_num_threads(1)
    oracle = MDNOracle(str(ROOT / "mdn_p_massive.pth"),
                       str(ROOT / "mdn_q_massive.pth"), torch.device("cpu"))
    case = {"bits": 32, "n": PARALLEL_P * PARALLEL_Q}
    return measure_solver(case, "mdn", oracle, max_steps, num_attempts, seed)


def parallel_runs(max_steps, num_attempts):
    runs = []
    context = multiprocessing.get_context("spawn")
    for workers in (1, 2, 4, 8):
        start = time.perf_counter()
        completions = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                                                      mp_context=context) as pool:
            future_to_seed = {
                pool.submit(parallel_worker, seed, max_steps, num_attempts): seed
                for seed in PARALLEL_SEEDS
            }
            for future in concurrent.futures.as_completed(future_to_seed):
                row = future.result()
                completions.append({"seed": row["seed"], "success": row["success"],
                                    "completed_s": time.perf_counter() - start,
                                    "solver_s": row["seconds"],
                                    "candidate_checks": row["candidate_checks"]})
        runs.append({
            "workers": workers, "n": PARALLEL_P * PARALLEL_Q,
            "seeds": list(PARALLEL_SEEDS), "completed": completions,
            "first_success_s": min((r["completed_s"] for r in completions if r["success"]),
                                   default=None),
            "all_done_s": time.perf_counter() - start,
            "success_count": sum(r["success"] for r in completions),
            "total_candidate_checks": sum(r["candidate_checks"] for r in completions),
        })
        print(f"workers={workers}: success={runs[-1]['success_count']}/8, "
              f"first={runs[-1]['first_success_s']}, all={runs[-1]['all_done_s']:.3f}s",
              flush=True)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--num-attempts", type=int, default=1000)
    parser.add_argument("--bit-sizes", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--exclude-cases", action="append", default=[],
                        help="JSON case file whose moduli must not reappear")
    parser.add_argument("--skip-parallel", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    archive = archived_moduli()
    excluded = set(archive)
    for file in args.exclude_cases:
        excluded.update(int(case["n"]) for case in json.loads(Path(file).read_text()))
    cases = [case for bits in args.bit_sizes
             for case in generate_cases(bits, args.samples_per_size, excluded, args.seed)]
    out = ROOT / args.output_dir
    out.mkdir(exist_ok=True)
    (out / "small_key_cases.json").write_text(json.dumps(cases, indent=2) + "\n")

    load_start = time.perf_counter()
    oracle = MDNOracle(str(ROOT / "mdn_p_massive.pth"),
                       str(ROOT / "mdn_q_massive.pth"), torch.device("cpu"))
    model_load_seconds = time.perf_counter() - load_start
    rows = []
    for i, case in enumerate(cases):
        seed = args.seed + i
        rows.append(measure_trial(case))
        rows.append(measure_solver(case, "no_jump", NoJumpOracle(),
                                   args.max_steps, args.num_attempts, seed))
        rows.append(measure_solver(case, "uniform_hyperbola", UniformHyperbolaOracle(),
                                   args.max_steps, args.num_attempts, seed))
        rows.append(measure_solver(case, "mdn", oracle,
                                   args.max_steps, args.num_attempts, seed))
        if (i + 1) % args.samples_per_size == 0:
            bits = case["bits"]
            relevant = [r for r in rows if r["bits"] == bits]
            print(f"{bits}-bit N: " + ", ".join(
                f"{mode}={sum(r['success'] for r in relevant if r['mode'] == mode)}/"
                f"{args.samples_per_size}"
                for mode in ("trial_division", "no_jump", "uniform_hyperbola", "mdn")),
                flush=True)

    metadata = {"seed": args.seed, "samples_per_size": args.samples_per_size,
                "bit_sizes": args.bit_sizes,
                "max_steps": args.max_steps, "num_attempts": args.num_attempts,
                "search_radius": 1000, "torch": torch.__version__,
                "sympy": sympy.__version__, "python": platform.python_version(),
                "model_load_seconds": model_load_seconds,
                "archive_test_moduli_excluded": len(archive),
                "total_moduli_excluded": len(excluded),
                "training_set_unknown": True}
    (out / "small_key_measurements.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n")
    print(f"model load: {model_load_seconds:.3f}s", flush=True)
    if not args.skip_parallel:
        runs = parallel_runs(args.max_steps, args.num_attempts)
        (out / "parallel_one_n.json").write_text(
            json.dumps({"metadata": metadata,
                        "case": {"n": PARALLEL_P * PARALLEL_Q,
                                 "p": PARALLEL_P, "q": PARALLEL_Q},
                        "runs": runs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
