import argparse
import csv
import json
import sys
import time
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None

import jax.numpy as jnp

from model import (
    ModelParams,
    constrained_branch_lengths,
    constrained_model_params,
    edge_order,
)
from myo import load_tape_states, params_to_json, print_tree, read_full_tree
from tree_search import tree_search_


def load_asymmetric_params(path: Path, dt_override: float | None = None) -> ModelParams:
    with path.open() as f:
        raw = json.load(f)

    return ModelParams(
        rho=float(raw["rho"]),
        eta=jnp.asarray(raw["eta"], dtype=jnp.float32),
        tau=float(raw["tau"]),
        lambd=float(raw["lambda"]),
        m=int(raw.get("m", raw.get("max_tape_len", len(raw["eta"])))),
        dt=float(raw.get("dt", 0.05) if dt_override is None else dt_override),
    )


def root_length_from_full_tree(path: Path) -> float:
    tree = read_full_tree(str(path))
    dummy_roots = [v for v in tree.nodes if tree.in_degree[v] == 0]
    if len(dummy_roots) != 1:
        raise ValueError(f"expected one dummy root in {path}, found {len(dummy_roots)}")

    dummy_root = dummy_roots[0]
    roots = list(tree.successors(dummy_root))
    if len(roots) != 1:
        raise ValueError(f"expected one true-root edge from dummy root in {path}")

    return float(tree[dummy_root][roots[0]].get("weight", 0.0))


def write_history(history, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "neg_log_likelihood"])
        for step, value in enumerate(history):
            writer.writerow([step, float(value)])


def config_best_tree(best_tree, edge_lengths, root_length):
    root = [v for v in best_tree.nodes if best_tree.in_degree[v] == 0][0]
    for i, (u, v) in enumerate(edge_order(best_tree)):
        best_tree[u][v]["weight"] = edge_lengths[i].item()

    best_tree.add_node(-1)
    best_tree.add_edge(-1, root, weight=root_length.item())
    return best_tree


def parse_nni_edges(value: str | None):
    if value is None or value == "auto":
        return "auto"
    if value.lower() == "none":
        return None
    return int(value)


def peak_rss_bytes() -> int | None:
    if resource is None:
        return None

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def default_metrics_path(args) -> Path:
    instance_dir = Path(args.simulation_dir) / str(args.instance)
    output_dir = Path(args.output_dir) if args.output_dir else instance_dir
    return output_dir / "tree_search_metrics.json"


def write_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run convergence-study tree search on one simulated instance."
    )
    parser.add_argument("instance", help="Instance id, e.g. 0 for simulated_data/0.")
    parser.add_argument("--simulation-dir", default="simulated_data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--final-steps", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--nni-edges", default="auto")
    parser.add_argument("--num-trees", type=int, default=5)
    parser.add_argument("--screen-top-k", type=int, default=5)
    parser.add_argument("--candidate-polish-steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Where to write timing and memory metrics JSON. Defaults to output-dir/tree_search_metrics.json.",
    )
    return parser


def run_tree_search_instance(args) -> dict:

    instance_dir = Path(args.simulation_dir) / str(args.instance)
    output_dir = Path(args.output_dir) if args.output_dir else instance_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, leaves = load_tape_states(str(instance_dir / "sampled_leaves.csv"))
    params = load_asymmetric_params(instance_dir / "asymmetric_params.json", args.dt)
    root_length = root_length_from_full_tree(instance_dir / "full_edgelist.csv")

    nni_edges = parse_nni_edges(args.nni_edges)
    if nni_edges == "auto":
        nni_edges = 15 if len(labels) > 25 else None

    name = args.name if args.name is not None else str(args.instance)
    best_tree, best_raw_params, _, history = tree_search_(
        labels,
        leaves,
        params,
        args.learning_rate,
        args.inner_steps,
        args.final_steps,
        args.grad_clip_norm,
        args.tolerance,
        root_length,
        n=args.num_trees,
        max_steps=args.max_steps,
        nni_edges=nni_edges,
        screen_top_k=args.screen_top_k,
        candidate_polish_steps=args.candidate_polish_steps,
        name=name,
    )

    reconstructed_params, reconstructed_root_length, _ = constrained_model_params(
        best_raw_params, params.m, params.dt
    )
    edge_lengths = constrained_branch_lengths(best_tree, best_raw_params)
    reconstructed_tree = config_best_tree(
        best_tree, edge_lengths, reconstructed_root_length
    )

    params_to_json(
        reconstructed_params, str(output_dir / "reconstructed_params.json")
    )
    print_tree(reconstructed_tree, str(output_dir / "reconstructed_tree.csv"))
    write_history(history, output_dir / "tree_search_history.csv")

    summary = {
        "instance": str(args.instance),
        "sampled_leaves": len(labels),
        "final_neg_log_likelihood": float(history[-1]),
        "output_dir": str(output_dir),
    }
    print(
        "finished",
        f"instance={summary['instance']}",
        f"sampled_leaves={summary['sampled_leaves']}",
        f"final_neg_log_likelihood={summary['final_neg_log_likelihood']}",
        f"output_dir={output_dir}",
    )
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    start_wall = time.perf_counter()
    start_process = time.process_time()
    start_peak_rss_bytes = peak_rss_bytes()
    summary = {}
    status = "ok"
    error = None

    try:
        summary = run_tree_search_instance(args)
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        end_peak_rss_bytes = peak_rss_bytes()
        peak_rss_delta_bytes = (
            None
            if start_peak_rss_bytes is None or end_peak_rss_bytes is None
            else max(0, end_peak_rss_bytes - start_peak_rss_bytes)
        )
        metrics_path = (
            Path(args.metrics_path) if args.metrics_path else default_metrics_path(args)
        )
        metrics = {
            "status": status,
            "error": error,
            "wall_seconds": time.perf_counter() - start_wall,
            "process_cpu_seconds": time.process_time() - start_process,
            "peak_rss_bytes": end_peak_rss_bytes,
            "peak_rss_mb": (
                None if end_peak_rss_bytes is None else end_peak_rss_bytes / (1024 * 1024)
            ),
            "peak_rss_delta_bytes": peak_rss_delta_bytes,
            "peak_rss_delta_mb": (
                None
                if peak_rss_delta_bytes is None
                else peak_rss_delta_bytes / (1024 * 1024)
            ),
            **summary,
        }
        try:
            write_metrics(metrics, metrics_path)
            print(f"metrics_path={metrics_path}")
        except OSError as metrics_error:
            print(f"warning: failed to write metrics: {metrics_error}", file=sys.stderr)


if __name__ == "__main__":
    main()
