from itertools import combinations
import networkx as nx
from myo import read_full_tree, load_tape_states, skeleton_tree
import numpy as np

import numpy as np
import matplotlib.pyplot as plt


def plot_distances(D, path):
    plt.imshow(D)
    plt.colorbar()
    plt.title("Pairwise distances")
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
    return reconstructed_distances, leaf_idx_map


def process_underlying_tree(edgelist_path, sampled_leaves_path, leaf_idx_map):
    tree = read_full_tree(edgelist_path)
    states, leaves = load_tape_states(sampled_leaves_path)
    dummy_root = [v for v in tree.nodes() if tree.in_degree(v) == 0][0]
    root = list(tree.successors(dummy_root))
    assert len(root) == 1, "Failed to remove dummy root"
    skeleton_T = skeleton_tree(tree, root[0], leaves)
    sampled_leaf_distances = tree_distance_map(skeleton_T, leaves, leaf_idx_map)
    return sampled_leaf_distances, states, leaves

def scale_distances(A):
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

if __name__ == "__main__":
    seed = 10
    true_tree = f"./simulations/{seed}/full_edgelist.csv"
    sampled_leaves = f"./simulations/{seed}/sampled_leaves.csv"
    reconstruted_tree = f"./simulations/{seed}/reconstructed_tree.csv"
    
    recon_D, leaf_idx_map = process_reconstructed_tree(reconstruted_tree)
    true_D, states, leaves = process_underlying_tree(true_tree, sampled_leaves, leaf_idx_map)

    plot_distances(recon_D, f"./simulations/{seed}/reconstructed_distances.png")
    plot_distances(true_D, f"./simulations/{seed}/true_distances.png")
    
    err = np.linalg.norm(recon_D - true_D) / np.linalg.norm(true_D)
    print(err)
    