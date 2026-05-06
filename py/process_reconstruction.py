from itertools import combinations
import json
import networkx as nx
from myo import read_full_tree, load_tape_states, skeleton_tree
import numpy as np
from tree_search import init_tree, tape_distances
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from model import ModelParams, TapeState, log_likelihood


def plot_distances(D, path, title):
    plt.imshow(D)
    plt.colorbar()
    plt.title(title)
    plt.savefig(path)
    plt.close()


def tree_distance_map(T: nx.DiGraph, leaves, leaf_idx_map):
    n = len(leaves)
    D = np.zeros((n, n), dtype=float)
    # turn into undirected
    G = nx.Graph(T)
    for u, v in combinations(leaves, 2):
        shortest_path = nx.shortest_path_length(G, source=u, target=v, weight="weight")
        i = leaf_idx_map[u]
        j = leaf_idx_map[v]
        if i > j:
            D[j, i] = shortest_path
        else:
            D[i, j] = shortest_path

    return D


def process_reconstructed_tree(edgelist_path):
    reconstructed_T = read_full_tree(edgelist_path)
    leaves = [u for u in reconstructed_T.nodes() if reconstructed_T.out_degree[u] == 0]
    leaf_idx_map = {l: i for i, l in enumerate(leaves)}
    reconstructed_distances = tree_distance_map(reconstructed_T, leaves, leaf_idx_map)
    return reconstructed_distances, leaf_idx_map, reconstructed_T


def process_underlying_tree(edgelist_path, sampled_leaves_path, leaf_idx_map):
    tree = read_full_tree(edgelist_path)
    states, leaves = load_tape_states(sampled_leaves_path)
    dummy_root = [v for v in tree.nodes() if tree.in_degree(v) == 0][0]
    root = list(tree.successors(dummy_root))
    assert len(root) == 1, "Failed to remove dummy root"
    leaves = [str(v) for v in leaves]
    skeleton_T = skeleton_tree(tree, root[0], leaves)
    for leaf, state in zip(leaves, states):
        skeleton_T.nodes[leaf]["label"] = state
    sampled_leaf_distances = tree_distance_map(skeleton_T, leaves, leaf_idx_map)
    return sampled_leaf_distances, states, leaves, skeleton_T, root[0]


def load_asymmetric_params(path, dt=0.05):
    with open(path) as f:
        raw = json.load(f)

    eta = jnp.asarray(raw["eta"], dtype=jnp.float32)
    return ModelParams(
        rho=float(raw["rho"]),
        eta=eta,
        tau=float(raw["tau"]),
        lambd=float(raw["lambda"]),
        m=int(raw.get("m", raw.get("max_tape_len", len(raw["eta"])))),
        dt=float(raw.get("dt", dt)),
    )


def root_length_from_full_tree(edgelist_path):
    tree = read_full_tree(edgelist_path)
    dummy_root = [v for v in tree.nodes() if tree.in_degree(v) == 0][0]
    root = list(tree.successors(dummy_root))
    assert len(root) == 1, "Expected one edge from dummy root to true root"
    return float(tree[dummy_root][root[0]].get("weight", 0.0))


def true_tree_neg_log_likelihood(true_t, true_tree_path, params_path, dt=0.05):
    params = load_asymmetric_params(params_path, dt=dt)
    root_length = root_length_from_full_tree(true_tree_path)
    logl = log_likelihood(true_t, params, root_length, {})
    return -float(logl), root_length, params


def normalize_distances(A):
    mask = A != 0

    nz = A[mask]
    A_min = nz.min()
    A_max = nz.max()

    if A_max == A_min:
        A_scaled = np.zeros_like(A)
        A_scaled[mask] = 1.0  # or keep original values
    else:
        A_scaled = A.copy().astype(float)
        A_scaled[mask] = (A[mask] - A_min) / (A_max - A_min)

    return A_scaled


def clades(T: nx.DiGraph, root):
    C = set()

    def dfs(node):
        if T.out_degree(node) == 0:
            return frozenset([node])

        below = frozenset().union(*(dfs(c) for c in T.successors(node)))

        if node != root:
            C.add(below)

        return below

    dfs(root)
    return C


def norm_rf_distance(T1: nx.DiGraph, t1_root, T2: nx.DiGraph, t2_root):
    t1_clades = clades(T1, t1_root)
    t2_clades = clades(T2, t2_root)
    assert T1.in_degree(t1_root) == 0
    assert T2.in_degree(t2_root) == 0

    t1_diff = t1_clades.difference(t2_clades)
    t2_diff = t2_clades.difference(t1_clades)

    # print("Same Clades")
    # for clade in t1_clades.intersection(t2_clades):
    #     print(clade)
    # print("T2")
    # for clade in t2_clades:
    #     print(clade)

    return (len(t1_diff) + len(t2_diff)) / (len(t1_clades) + len(t2_clades))


def test_splits():
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
    tree = nx.DiGraph()
    for u, v, w in edges:
        tree.add_node(u)
        tree.add_node(v)
        tree.add_edge(u, v, weight=w)

    print(clades(tree, 0))


def test_distances():
    simulations_dir = "simulated_data"
    seed = 0
    true_tree = f"./{simulations_dir}/{seed}/full_edgelist.csv"
    sampled_leaves = f"./{simulations_dir}/{seed}/sampled_leaves.csv"
    params_path = f"./{simulations_dir}/{seed}/asymmetric_params.json"
    reconstruted_tree = f"./{simulations_dir}/{seed}/reconstructed_tree.csv"
    # reconstruted_tree = f".{seed}_so_far_reconstructed_tree.csv"

    recon_D, leaf_idx_map, recon_t = process_reconstructed_tree(reconstruted_tree)
    recon_root = [v for v in recon_t.nodes() if recon_t.in_degree[v] == 0][0]

    true_D, states, leaves, true_t, true_root = process_underlying_tree(
        true_tree, sampled_leaves, leaf_idx_map
    )

    rf_distance = norm_rf_distance(recon_t, recon_root, true_t, true_root)
    print("Normalized RF distance:", rf_distance)

    # true_nll, true_root_length, true_params = true_tree_neg_log_likelihood(
    #     true_t, true_tree, params_path
    # )
    # print(
    #     "true tree neg-log-likelihood:",
    #     true_nll,
    #     "root_length:",
    #     true_root_length,
    #     "params:",
    #     true_params,
    # )

    # plot_distances(normalize_distances(recon_D), f"./{simulations_dir}/{seed}/normalized_reconstructed_distances.png", "Reconstructed Leaf Normalized Distances")
    # plot_distances(normalize_distances(true_D), f"./{simulations_dir}/{seed}/normalized_true_distances.png", "True Leaf Normalized Distances")
    # plot_distances(
    #     dist_D,
    #     f"./{simulations_dir}/{seed}/distance_tree_distances.png",
    # )

    err = np.linalg.norm(recon_D - true_D) / np.linalg.norm(true_D)
    print("Normalized L_2 norm of leaf distance matrices:", err)


if __name__ == "__main__":
    test_distances()
