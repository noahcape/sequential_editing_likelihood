import itertools
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
import networkx as nx
from model import (
    TapeState,
    ModelParams,
    initial_height_increments,
    positive_inverse_transform,
)
from n_model import (
    array_neg_log_likelihood_from_raw_params,
    constrain_array_raw_params,
    encode_tree,
    init_array_raw_params,
    initial_height_increments_by_pos,
    optimize_array_likelihood,
)
import jax.numpy as jnp
import random
from itertools import combinations
import numpy as np
from myo import params_to_json, print_tree

def tape_distances(labels, leaves):
    leaves = list(leaves)
    labels = list(labels)

    n = len(leaves)
    D = np.zeros((n, n), dtype=float)

    for a, b in combinations(range(n), 2):
        distance = TapeState.distance(labels[a], labels[b])
        D[a, b] = distance
        D[b, a] = distance

    return D

def init_tree(labels, leaves, temp=0.0, rng=None, idx_map=None):
    leaves = list(leaves)
    labels = list(labels)
    label_by_leaf = dict(zip(leaves, labels))

    D = np.asarray(tape_distances(labels, leaves), dtype=float)

    condensed = squareform(D, checks=False)
    if temp is not None and temp > 0.0:
        rng = np.random.default_rng() if rng is None else rng
        condensed = np.maximum(
            condensed + rng.gumbel(loc=0.0, scale=temp, size=condensed.shape),
            0.0,
        )

    Z = linkage(condensed, method="average")
    root_cluster = to_tree(Z)

    reverse_map = {i: leaf for i, leaf in enumerate(leaves)}

    T = nx.DiGraph()
    next_internal = itertools.count(len(leaves))

    def add_node(cluster):
        if cluster.is_leaf():
            leaf = reverse_map[cluster.id]
            T.add_node(leaf, label=label_by_leaf[leaf])
            return leaf

        u = f"internal_{next(next_internal)}"
        T.add_node(u)

        left = add_node(cluster.left)
        right = add_node(cluster.right)

        left_len = max(cluster.dist - cluster.left.dist, 0.0)
        right_len = max(cluster.dist - cluster.right.dist, 0.0)

        T.add_edge(u, left, weight=left_len)
        T.add_edge(u, right, weight=right_len)

        return u

    root = add_node(root_cluster)
    T.graph["root"] = root
    return T

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
        (u, v) for (u, v) in T.edges() if T.out_degree[u] == 2 and T.out_degree[v] == 2
    ]


def is_internal(T, u, v):
    return T.out_degree[u] == 2 and T.out_degree[v] == 2


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

    to_apply_nnis = min(10, int(0.5 * len(internal_edges)))
    applied_nnis = 0
    mixed_T = T.copy()
    # now modify the original tree with max(10, 1/2 internal edges) nnis
    while applied_nnis < to_apply_nnis:
        u,v = random.sample(list(mixed_T.edges()), 1)[0]
        if is_internal(mixed_T, u, v):
            u_side = [x for x in mixed_T.successors(u) if x != v]
            v_side = list(mixed_T.successors(v))

            b = u_side[0]
            c, d = v_side

            ub_w = mixed_T.get_edge_data(u, b)["weight"]
            vc_w = mixed_T.get_edge_data(v, c)["weight"]

            mixed_T.remove_edge(u, b)
            mixed_T.remove_edge(v, c)
            mixed_T.add_edge(u, c, weight=ub_w)
            mixed_T.add_edge(v, b, weight=vc_w)

            applied_nnis += 1
        else:
            continue

    nnis.append(mixed_T)
    return nnis


def sync_tree_branch_lengths(T: nx.DiGraph, raw_params, m: int, dt: float) -> None:
    """Copy optimized branch lengths from packed params back onto the tree edges."""
    encoded = encode_tree(T, ModelParams(0.1, raw_params["eta"], 1.0, 1.0, m, dt))
    _, _, branch_lengths = constrain_array_raw_params(encoded, raw_params, dt, m)
    for pos, node in enumerate(encoded.postorder_nodes):
        parent_pos = int(encoded.parent[pos])
        if parent_pos >= 0:
            parent = encoded.postorder_nodes[parent_pos]
            T[parent][node]["weight"] = float(branch_lengths[pos])


def reinitialize_array_topology_params(encoded, raw_params):
    """Keep model rates/root length and reinitialize heights for this topology."""
    candidate_params = dict(raw_params)
    candidate_params["height_increments_by_pos"] = positive_inverse_transform(
        initial_height_increments_by_pos(encoded)
    )
    return candidate_params


def array_raw_to_model_raw(tree: nx.DiGraph, raw_params, params_: ModelParams):
    """Convert array optimizer params back to the legacy raw-parameter layout."""
    sync_tree_branch_lengths(tree, raw_params, params_.m, params_.dt)
    return {
        "rho": raw_params["rho"],
        "eta": raw_params["eta"],
        "tau": raw_params["tau"],
        "lambd": raw_params["lambd"],
        "root_length": raw_params["root_length"],
        "height_increments": positive_inverse_transform(initial_height_increments(tree)),
    }


def tree_with_root_edge(
    tree: nx.DiGraph, raw_params, params_: ModelParams
) -> nx.DiGraph:
    """Return a copy of tree with optimized lengths and the external root edge."""
    rooted_tree = tree.copy()
    sync_tree_branch_lengths(rooted_tree, raw_params, params_.m, params_.dt)
    encoded = encode_tree(rooted_tree, params_)
    _, root_length, _ = constrain_array_raw_params(
        encoded, raw_params, params_.dt, params_.m
    )
    root = [v for v in rooted_tree.nodes if rooted_tree.in_degree[v] == 0][0]
    rooted_tree.add_node(-1)
    rooted_tree.add_edge(-1, root, weight=root_length.item())
    return rooted_tree


def max_likelihood_tree_search(
    T: nx.DiGraph,
    raw_params,
    params_: ModelParams,
    learning_rate,
    optimization_loop_steps,
    grad_clip_norm,
    tol,
    max_steps=10,
    nni_edges=None,
    screen_top_k=5,
    candidate_polish_steps=5,
    jit_optimizations=False,
):
    root = [v for v in T.nodes if T.in_degree[v] == 0]
    assert len(root) == 1, "Must have unique root"

    raw_params_iter = raw_params
    encoded = encode_tree(T, params_)
    ml = array_neg_log_likelihood_from_raw_params(
        encoded, raw_params_iter, params_.dt
    )
    step_likelihood = [ml]

    steps = 0
    while steps < max_steps:
        sync_tree_branch_lengths(T, raw_params_iter, params_.m, params_.dt)
        nnis = nni_neighborhood(T, n=nni_edges)

        print("current best likelihood:", ml)
        scored_nnis = []
        for nT in nnis:
            # Reinitialize the ultrametric height variables for this topology.
            candidate_encoded = encode_tree(nT, params_)
            candidate_params = reinitialize_array_topology_params(
                candidate_encoded, raw_params_iter
            )
            this_likel = array_neg_log_likelihood_from_raw_params(
                candidate_encoded, candidate_params, params_.dt
            )
            print("nnis likelihood:", this_likel)
            scored_nnis.append(
                (float(this_likel), nT, candidate_encoded, candidate_params)
            )

        scored_nnis.sort(key=lambda x: x[0])
        shortlist = scored_nnis[: min(screen_top_k, len(scored_nnis))]
        best_nni = (T, ml, dict(raw_params_iter))

        for cheap_likel, nT, candidate_encoded, candidate_params in shortlist:
            if candidate_polish_steps > 0:
                polished_params, polish_history = optimize_array_likelihood(
                    candidate_encoded,
                    candidate_params,
                    params_.dt,
                    learning_rate,
                    candidate_polish_steps,
                    grad_clip_norm,
                )
                polished_likel = array_neg_log_likelihood_from_raw_params(
                    candidate_encoded, polished_params, params_.dt
                )
            else:
                polished_params = candidate_params
                polished_likel = cheap_likel

            print("polished nni likelihood:", polished_likel)
            if polished_likel < best_nni[1]:
                best_nni = (nT, polished_likel, polished_params)

        (T, t_ml, raw_params_iter) = best_nni

        # break condition
        # add a simulated annealing step here weighted by the gain multiplied by the temp
        if abs(ml - t_ml) < tol:
            break

        # optimize at each iteration
        encoded = encode_tree(T, params_)
        raw_params_iter, history = optimize_array_likelihood(
            encoded,
            raw_params_iter,
            params_.dt,
            learning_rate,
            optimization_loop_steps,
            grad_clip_norm,
        )
        print(history)

        ml = array_neg_log_likelihood_from_raw_params(encoded, raw_params_iter, params_.dt)
        steps += 1
        step_likelihood.append(ml)

    sync_tree_branch_lengths(T, raw_params_iter, params_.m, params_.dt)
    return T, raw_params_iter, step_likelihood, None


def temp_perturbed_agglom_initial_trees(
    labels,
    leaves,
    n=10,
    include_agglomerative=True,
    agglomerative_temperature=0.5,
):
    if not include_agglomerative:
        return [random_tree(labels, leaves) for _ in range(n)]

    rng = np.random.default_rng()
    trees = [init_tree(labels, leaves, temp=0.0, rng=rng)]
    trees.extend(
        init_tree(labels, leaves, temp=agglomerative_temperature, rng=rng)
        for _ in range(max(n - 1, 0))
    )
    return trees


def tree_search_(
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
    nni_edges=None,
    screen_top_k=5,
    candidate_polish_steps=5,
    include_agglomerative=True,
    agglomerative_temperature=0.5,
    jit_optimizations=False,
    name="",
):
    initial_random_trees = temp_perturbed_agglom_initial_trees(
        labels, leaves, n, include_agglomerative, agglomerative_temperature
    )
    best_ml = jnp.inf
    best_T = None
    best_params = None
    best_ts_history = []

    # do this in parallel
    for init_t in initial_random_trees:
        init_encoded = encode_tree(init_t, params_)
        raw_params = init_array_raw_params(
            init_encoded,
            params_.eta,
            params_.rho,
            params_.tau,
            params_.lambd,
            root_length,
        )
        T, params, history, _ = max_likelihood_tree_search(
            init_t,
            raw_params,
            params_,
            learning_rate,
            optimization_loop_steps,
            grad_clip_norm,
            tol,
            max_steps,
            nni_edges,
            screen_top_k,
            candidate_polish_steps,
            jit_optimizations,
        )
        ml = history[-1]
        if ml < best_ml:
            best_ml = ml
            best_T = T
            best_params = params
            best_ts_history = history

            # temp write best likelihood tree
            best_encoded = encode_tree(best_T, params_)
            model_params, _, _ = constrain_array_raw_params(
                best_encoded, best_params, params_.dt, params_.m
            )

            params_to_json(model_params, f"./{name}_so_far_reconstructed_params.json")

            best_tree = tree_with_root_edge(best_T, best_params, params_)
            print_tree(best_tree, f".{name}_so_far_reconstructed_tree.csv")

    best_encoded = encode_tree(best_T, params_)
    params, history = optimize_array_likelihood(
        best_encoded,
        best_params,
        params_.dt,
        learning_rate,
        final_steps,
        grad_clip_norm,
    )
    sync_tree_branch_lengths(best_T, params, params_.m, params_.dt)
    legacy_params = array_raw_to_model_raw(best_T, params, params_)
    final_ml = array_neg_log_likelihood_from_raw_params(
        encode_tree(best_T, params_), params, params_.dt
    )

    return best_T, legacy_params, final_ml, best_ts_history + history + [final_ml]


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
    params = ModelParams(
        0.1, jnp.asarray([0.1, 0.1, 0.1], dtype=float), 0.2, 1.0, 3, 0.5
    )
    raw = init_array_raw_params(
        encode_tree(T, params),
        eta=params.eta,
        rho=params.rho,
        tau=params.tau,
        lambd=params.lambd,
        root_length=1.5,
    )

    _, raw_opt, ml, _ = max_likelihood_tree_search(
        T,
        raw,
        params,
        learning_rate=1e-2,
        optimization_loop_steps=1,
        grad_clip_norm=1.0,
        tol=1e-5,
        max_steps=1,
        nni_edges=1,
    )

    print(ml)
    print(raw_opt)


if __name__ == "__main__":
    build_test_tree()
