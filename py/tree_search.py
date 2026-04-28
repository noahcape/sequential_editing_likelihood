import networkx as nx
from model import (
    TapeState,
    ModelParams,
    constrained_model_params,
    constrained_branch_lengths,
    init_raw_params,
    initial_height_increments,
    neg_log_likelihood_from_raw_params,
    optimize_likelihood,
    edge_order,
    populate_tape_graphs,
    ensure_fixed_tape_graphs,
    positive_inverse_transform,
)
import jax.numpy as jnp
import random

def random_tree(labels, leaves):
    """
    Initialize a random tree with n leaves where n = len(labels)
    Initialize each each length to be 5.0
    """
    tree = nx.DiGraph()
    for i, label in zip(leaves, labels):
        tree.add_node(i, label=label)

    current_nodes = leaves.copy()
    idx = max(leaves) + 1
    # randomly sample two nodes and join them
    while len(current_nodes) > 1:
        a, b = random.sample(current_nodes, 2)
        current_nodes.remove(a)
        current_nodes.remove(b)

        tree.add_node(idx)
        tree.add_edge(idx, a, weight=5.0)
        tree.add_edge(idx, b, weight=5.0)
        current_nodes.append(idx)
        idx += 1

    return tree


def get_internal_edges(T: nx.DiGraph):
    return [
        (u, v)
        for (u, v) in T.edges()
        if T.out_degree[u] == 2 and T.out_degree[v] == 2
    ]


def nni_neighborhood(T: nx.DiGraph, n=None):
    internal_edges = get_internal_edges(T)
    nnis = []
    
    # if n is set randomly sampled n internal edges to create nnis on
    if n is not None:
        internal_edges = random.sample(internal_edges, n)

    for u, v in internal_edges:
        # Neighbors on each side excluding the central edge
        u_side = [x for x in T.successors(u) if x != v]
        v_side = list(T.successors(v))

        b = u_side[0]
        c, d = v_side

        ub_w = T.get_edge_data(u, b)["weight"]
        vc_w = T.get_edge_data(v, c)["weight"]
        vd_w = T.get_edge_data(v, d)["weight"]

        # First nni: swap b and c
        T1 = T.copy()
        T1.remove_edge(u, b)
        T1.remove_edge(v, c)
        T1.add_edge(u, c, weight=ub_w)
        T1.add_edge(v, b, weight=vc_w)
        nnis.append(T1)

        # Second nni: swap b and d
        T2 = T.copy()
        T2.remove_edge(u, b)
        T2.remove_edge(v, d)
        T2.add_edge(u, d, weight=ub_w)
        T2.add_edge(v, b, weight=vd_w)
        nnis.append(T2)

    return nnis


def sync_tree_branch_lengths(T: nx.DiGraph, raw_params, m: int, dt: float) -> None:
    """Copy optimized branch lengths from packed params back onto the tree edges."""
    branch_lengths = constrained_branch_lengths(T, raw_params)
    edges = edge_order(T)
    assert branch_lengths.shape[0] == len(edges)
    for i, (u, v) in enumerate(edges):
        T[u][v]["weight"] = float(branch_lengths[i])


def max_likelihood_tree_search(
    T: nx.DiGraph,
    raw_params,
    params_: ModelParams,
    learning_rate,
    optimization_loop_steps,
    grad_clip_norm,
    tol,
    max_steps=10,
    nni_edges=None
):
    tape_graphs = {}
    root = [v for v in T.nodes if T.in_degree[v] == 0]
    assert len(root) == 1, "Must have unique root"

    populate_tape_graphs(T, root[0], tape_graphs, params_)
    ensure_fixed_tape_graphs(tape_graphs, params_)

    raw_params_iter = raw_params
    ml = neg_log_likelihood_from_raw_params(
        T, raw_params_iter, params_.m, params_.dt, tape_graphs
    )
    step_likelihood = [ml]

    steps = 0
    while True:
        sync_tree_branch_lengths(T, raw_params_iter, params_.m, params_.dt)
        best_nni = (T, ml, dict(raw_params_iter))
        nnis = nni_neighborhood(T, n=nni_edges)
        for nT in nnis:
            root = [v for v in nT.nodes if nT.in_degree[v] == 0]
            assert len(root) == 1, "Must have unique root"
            populate_tape_graphs(nT, root[0], tape_graphs, params_)
        ensure_fixed_tape_graphs(tape_graphs, params_)

        print("current best likelihood:", ml)
        for nT in nnis:
            # Reinitialize the ultrametric height variables for this topology.
            height_increments = initial_height_increments(nT)
            candidate_params = dict(raw_params_iter)
            candidate_params["height_increments"] = positive_inverse_transform(height_increments)  # type: ignore
            this_likel = neg_log_likelihood_from_raw_params(
                nT, candidate_params, params_.m, params_.dt, tape_graphs
            )
            print("nnis likelihood:", this_likel)
            if this_likel < best_nni[1]:
                best_nni = (nT, this_likel, candidate_params)

        (T, t_ml, raw_params_iter) = best_nni

        # break condition
        if abs(ml - t_ml) < tol or steps == max_steps:
            break

        # optimize at each iteration
        raw_params_iter, history = optimize_likelihood(
            T,
            raw_params_iter,
            tape_graphs,
            params_.m,
            params_.dt,
            learning_rate,
            optimization_loop_steps,
            grad_clip_norm,
        )
        print(history)

        ml = history[-1]
        steps += 1
        step_likelihood.append(ml)

    sync_tree_branch_lengths(T, raw_params_iter, params_.m, params_.dt)
    return T, raw_params_iter, step_likelihood, tape_graphs


def random_initial_trees(labels, leaves, n=10):
    return [random_tree(labels, leaves) for _ in range(n)]


def tree_search(
    labels,
    leaves,
    params_: ModelParams,
    learning_rate,
    optimization_loop_steps,
    final_steps,
    grad_clip_norm,
    tol,
    root_length,
    n=10,
    max_steps=10,
    nni_edges=None
):
    initial_random_trees = random_initial_trees(labels, leaves, n)

    best_ml = jnp.inf
    best_T = None
    best_params = None
    best_ts_history = []
    best_tape_graphs = {}

    # do this in parallel
    for init_t in initial_random_trees:
        raw_params = init_raw_params(
            init_t, params_.eta, params_.rho, params_.tau, params_.lambd, root_length
        )
        T, params, history, tape_graphs = max_likelihood_tree_search(
            init_t,
            raw_params,
            params_,
            learning_rate,
            optimization_loop_steps,
            grad_clip_norm,
            tol,
            max_steps,
            nni_edges,
        )
        ml = history[-1]
        if ml < best_ml:
            best_ml = ml
            best_T = T
            best_params = params
            best_ts_history = history
            best_tape_graphs = tape_graphs

    params, history = optimize_likelihood(
        best_T,
        best_params,
        best_tape_graphs,
        params_.m,
        params_.dt,
        learning_rate,
        final_steps,
        grad_clip_norm,
    )
    sync_tree_branch_lengths(best_T, params, params_.m, params_.dt)

    return best_T, params, history[-1], best_ts_history + history


def build_test_tree() -> nx.DiGraph:
    edges = [
        (0, 1, 1.2),
        (0, 2, 0.8),
        (1, 3, 0.2),
        (1, 4, 2.4),
        (2, 5, 0.9),
        (2, 6, 1.1),
        (3, 7, 1.1),
        (3, 8, 1.1),
        (4, 9, 1.1),
        (4, 10, 1.1),
        (5, 11, 1.1),
        (5, 12, 1.1),
        (6, 13, 1.1),
        (6, 14, 1.1),
    ]
    labeling = [
        (7, TapeState((0, 1), (0,), 1)),
        (9, TapeState((0, 1, 1), (0, 1), 1)),
        (8, TapeState((1, 2), (1,), 1)),
        (10, TapeState((1, 2), (1, 2), 0)),
        (11, TapeState((2, 1), (2,), 1)),
        (12, TapeState((2, 0), (2, 0), 0)),
        (13, TapeState((0, 1), (0,), 0)),
        (14, TapeState((1,), (), 1)),
    ]
    tree = nx.DiGraph()
    for v, label in labeling:
        tree.add_node(v, label=label)

    for u, v, w in edges:
        tree.add_edge(u, v, weight=w)

    return tree


def test_tree_search():
    T = build_test_tree()

    raw = init_raw_params(
        T,
        eta=[0.1, 0.1, 0.1],
        rho=0.1,
        tau=0.2,
        lambd=1.0,
        root_length=1.5,
    )
    params = ModelParams(
        0.1, jnp.asarray([0.1, 0.1, 0.1], dtype=float), 0.2, 1.0, 3, 0.5
    )

    _, raw_opt, ml, _ = max_likelihood_tree_search(
        T, raw, params, 3, 0.5, 1e-2, 100, 1000, 1.0, 1e-5, 3
    )

    print(ml)
    print(constrained_model_params(raw_opt, 3, 0.5))


if __name__ == "__main__":
    build_test_tree()
