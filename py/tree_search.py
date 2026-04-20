import networkx as nx
from model import (
    TapeState,
    ModelParams,
    constrained_model_params,
    init_raw_params,
    neg_log_likelihood_from_raw_params,
    optimize_likelihood,
    edge_order,
    populate_tape_graphs,
)
import jax.numpy as jnp


def get_internal_edges(T: nx.DiGraph):
    return [(u, v) for (u, v) in T.edges() if T.degree[v] > 2 and T.degree[u] > 2]


def nni_neighborhood(T: nx.DiGraph):
    internal_edges = get_internal_edges(T)
    nnis = []

    for u, v in internal_edges:
        # Neighbors on each side excluding the central edge
        u_side = [x for x in T.neighbors(u) if x != v]
        v_side = [x for x in T.neighbors(v) if x != u]

        b = u_side[0]
        c, d = v_side

        ub_w = T.get_edge_data(u, b)["weight"]
        vc_w = T.get_edge_data(u, b)["weight"]
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


def max_likelihood_tree_search(
    T: nx.DiGraph,
    raw_params,
    params_: ModelParams,
    m,
    dt,
    learning_rate,
    optimization_loop_steps,
    final_steps,
    grad_clip_norm,
    tol,
    max_steps=10,
):
    tape_graphs = {}
    populate_tape_graphs(T, 0, tape_graphs, params_)

    params = raw_params
    ml = neg_log_likelihood_from_raw_params(T, params, m, dt, tape_graphs)

    steps = 0
    while True:
        best_nni = (T, ml, params)

        print("current best likelihood:", ml)
        for nT in nni_neighborhood(T):
            populate_tape_graphs(T, 0, tape_graphs, params_)
            # set branch lengths in params
            branch_lengths = jnp.array(
                [nT.get_edge_data(u, v)["weight"] for u, v in edge_order(nT)],
                dtype=jnp.float32,
            )
            params["branch_lengths"] = branch_lengths  # type: ignore
            this_likel = neg_log_likelihood_from_raw_params(
                nT, params, m, dt, tape_graphs
            )
            print("nnis likelihood:", this_likel)
            if this_likel < ml:
                best_nni = (nT, this_likel, params)

        (T, _, params) = best_nni
        # optimize at each iteration
        params, history = optimize_likelihood(
            T,
            raw_params,
            tape_graphs,
            m,
            dt,
            learning_rate,
            optimization_loop_steps,
            grad_clip_norm,
        )
        t_ml = history[-1]

        if abs(ml - t_ml) < tol or steps == max_steps:
            break

        ml = t_ml
        steps += 1

    params, history = optimize_likelihood(
        T, params, tape_graphs, m, dt, learning_rate, final_steps, grad_clip_norm
    )

    return T, params, history[-1]


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
        (8, TapeState((0, 1, 1), (0, 1), 1)),
        (9, TapeState((1, 2), (1,), 1)),
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

    _, raw_opt, ml = max_likelihood_tree_search(
        T, raw, params, 3, 0.5, 1e-2, 100, 1000, 1.0, 1e-5, 3
    )

    print(ml)
    print(constrained_model_params(raw_opt, 3, 0.5))


if __name__ == "__main__":
    test_tree_search()
