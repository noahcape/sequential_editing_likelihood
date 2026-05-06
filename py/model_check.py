import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import networkx as nx
import optax

from model import ModelParams, positive_inverse_transform, positive_transform
from myo import load_tape_states, read_full_tree, skeleton_tree
from n_model import (
    array_log_likelihood_with_branch_lengths,
    array_neg_log_likelihood_from_raw_params,
    array_ultrametric_branch_lengths,
    constrain_array_raw_params,
    encode_tree,
    init_array_raw_params,
    initial_height_increments_by_pos,
    optimize_array_likelihood,
)


def load_asymmetric_params(path: str | Path, dt: float = 0.05) -> ModelParams:
    with open(path) as f:
        raw = json.load(f)

    return ModelParams(
        rho=float(raw["rho"]),
        eta=jnp.asarray(raw["eta"], dtype=jnp.float32),
        tau=float(raw["tau"]),
        lambd=float(raw["lambda"]),
        m=int(raw.get("m", raw.get("max_tape_len", len(raw["eta"])))),
        dt=float(raw.get("dt", dt)),
    )


def true_root_and_length(tree: nx.DiGraph) -> tuple[Any, float]:
    dummy_root = [v for v in tree.nodes if tree.in_degree(v) == 0][0]
    roots = list(tree.successors(dummy_root))
    assert len(roots) == 1, "Expected one edge from dummy root to true root"
    root = roots[0]
    return root, float(tree[dummy_root][root].get("weight", 0.0))


def sampled_true_skeleton(
    tree_path: str | Path, sampled_leaves_path: str | Path
) -> tuple[nx.DiGraph, float]:
    full_tree = read_full_tree(str(tree_path))
    root, root_length = true_root_and_length(full_tree)
    states, leaves = load_tape_states(str(sampled_leaves_path))

    leaves = [str(leaf) for leaf in leaves]
    true_t = skeleton_tree(full_tree, root, leaves)
    for leaf, state in zip(leaves, states):
        true_t.nodes[leaf]["label"] = state

    return true_t, root_length


def branch_length_raw_params(encoded, root_length: float) -> dict[str, jnp.ndarray]:
    return {
        "root_length": positive_inverse_transform(
            jnp.asarray(root_length, dtype=jnp.float32)
        ),
        "height_increments_by_pos": positive_inverse_transform(
            initial_height_increments_by_pos(encoded)
        ),
    }


def root_to_leaf_depths(tree: nx.DiGraph, root_length: float) -> list[float]:
    root = [v for v in tree.nodes if tree.in_degree(v) == 0][0]
    leaves = [v for v in tree.nodes if tree.out_degree(v) == 0]
    depths = []
    for leaf in leaves:
        path = nx.shortest_path(tree, root, leaf)
        depth = root_length
        for u, v in zip(path, path[1:]):
            depth += float(tree[u][v].get("weight", 0.0))
        depths.append(depth)
    return depths


def branch_length_neg_log_likelihood(
    encoded,
    params: ModelParams,
    raw_params: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    root_length = positive_transform(
        jnp.asarray(raw_params["root_length"], dtype=jnp.float32)
    )
    height_increments = positive_transform(
        jnp.asarray(raw_params["height_increments_by_pos"], dtype=jnp.float32)
    )
    branch_lengths = array_ultrametric_branch_lengths(encoded, height_increments)
    return -array_log_likelihood_with_branch_lengths(
        encoded, params, root_length, branch_lengths
    )


def optimize_branch_lengths(
    encoded,
    params: ModelParams,
    raw_params: Mapping[str, jnp.ndarray],
    learning_rate: float,
    steps: int,
    grad_clip_norm: float,
) -> tuple[Mapping[str, jnp.ndarray], list[float]]:
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(raw_params)

    def loss_fn(x):
        return branch_length_neg_log_likelihood(encoded, params, x)

    @jax.jit
    def step_fn(x, state):
        loss, grads = jax.value_and_grad(loss_fn)(x)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=1.0e3, neginf=-1.0e3),
            grads,
        )
        updates, state = optimizer.update(grads, state, x)
        return optax.apply_updates(x, updates), state, loss

    history = []
    x = raw_params
    for _ in range(steps):
        x, opt_state, loss = step_fn(x, opt_state)
        history.append(float(loss))

    return x, history


def summarize_model_params(params: ModelParams) -> dict[str, Any]:
    return {
        "rho": float(params.rho),
        "eta": [float(x) for x in jnp.asarray(params.eta)],
        "tau": float(params.tau),
        "lambda": float(params.lambd),
        "m": int(params.m),
        "dt": float(params.dt),
    }


def run_check(
    seed: int,
    simulation_dir: str | Path,
    dt: float,
    branch_steps: int,
    all_steps: int,
    learning_rate: float,
    grad_clip_norm: float,
) -> None:
    sim_dir = Path(simulation_dir) / str(seed)
    true_tree_path = sim_dir / "full_edgelist.csv"
    sampled_leaves_path = sim_dir / "sampled_leaves.csv"
    params_path = sim_dir / "asymmetric_params.json"

    true_t, true_root_length = sampled_true_skeleton(
        true_tree_path, sampled_leaves_path
    )
    true_params = load_asymmetric_params(params_path, dt=dt)
    encoded = encode_tree(true_t, true_params)

    baseline_nll = -float(
        array_log_likelihood_with_branch_lengths(
            encoded, true_params, true_root_length, encoded.branch_lengths
        )
    )
    print(f"true topology + true params NLL: {baseline_nll:.6f}")
    print(f"true root_length: {true_root_length:.6f}")
    print(f"true params: {summarize_model_params(true_params)}")
    depths = root_to_leaf_depths(true_t, true_root_length)
    print(
        "true root-to-leaf depth range:",
        f"min={min(depths):.6f}",
        f"max={max(depths):.6f}",
        f"spread={max(depths) - min(depths):.6f}",
    )

    length_raw = branch_length_raw_params(encoded, true_root_length)
    ultrametric_initial_nll = float(
        branch_length_neg_log_likelihood(encoded, true_params, length_raw)
    )
    print(
        "true topology + ultrametric-projected lengths + true params NLL:",
        f"{ultrametric_initial_nll:.6f}",
    )
    length_raw, length_history = optimize_branch_lengths(
        encoded,
        true_params,
        length_raw,
        learning_rate=learning_rate,
        steps=branch_steps,
        grad_clip_norm=grad_clip_norm,
    )
    length_only_nll = float(
        branch_length_neg_log_likelihood(encoded, true_params, length_raw)
    )
    length_root = float(positive_transform(length_raw["root_length"]))
    print(f"true topology + optimized branch/root lengths NLL: {length_only_nll:.6f}")
    print(f"optimized root_length with true params: {length_root:.6f}")
    if length_history:
        print(
            "branch/root-only optimization:",
            f"start={length_history[0]:.6f}",
            f"end={length_history[-1]:.6f}",
        )

    all_raw = init_array_raw_params(
        encoded,
        true_params.eta,
        true_params.rho,
        true_params.tau,
        true_params.lambd,
        true_root_length,
    )
    all_raw, all_history = optimize_array_likelihood(
        encoded,
        all_raw,
        true_params.dt,
        learning_rate=learning_rate,
        steps=all_steps,
        grad_clip_norm=grad_clip_norm,
    )
    all_nll = float(array_neg_log_likelihood_from_raw_params(encoded, all_raw, true_params.dt))
    fitted_params, fitted_root_length, _ = constrain_array_raw_params(
        encoded, all_raw, true_params.dt, true_params.m
    )
    print(f"true topology + optimized model/branch/root NLL: {all_nll:.6f}")
    print(f"optimized root_length with fitted params: {float(fitted_root_length):.6f}")
    print(f"fitted params: {summarize_model_params(fitted_params)}")
    if all_history:
        print(
            "full fixed-topology optimization:",
            f"start={all_history[0]:.6f}",
            f"end={all_history[-1]:.6f}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether true simulation parameters are recovered on the true sampled skeleton."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simulation-dir", default="./simulated_data")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--branch-steps", type=int, default=100)
    parser.add_argument("--all-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    args = parser.parse_args()

    run_check(
        seed=args.seed,
        simulation_dir=args.simulation_dir,
        dt=args.dt,
        branch_steps=args.branch_steps,
        all_steps=args.all_steps,
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
    )


if __name__ == "__main__":
    main()
