"""
This file will test the convergence of the log-likelihood
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from myo import load_tape_states, print_tree, params_to_json
from model import (
    ModelParams,
    constrained_branch_lengths,
    constrained_model_params,
    edge_order,
    ensure_fixed_tape_graphs,
    optimize_likelihood,
    populate_tape_graphs,
)
from tree_search import tree_search



def plot_convergence_single(history, title, fname):
    x = np.arange(0, len(history))
    plt.figure()
    plt.plot(x, history, marker="o")
    plt.xlabel("Steps")
    plt.ylabel("-log(likelihood)")
    plt.title(title)
    plt.savefig(fname)


def plot_together(history_fixed, history_tree_search, title, fname):
    x_fixed = np.arange(0, len(history_fixed))
    x_ts = np.arange(0, len(history_tree_search))
    plt.figure()
    plt.plot(x_fixed, history_fixed, marker="o", color="r", legend="Fixed Tree")
    plt.plot(x_ts, history_tree_search, marker="o", color="b", legend="Tree Search")
    plt.xlabel("Steps")
    plt.ylabel("-log(likelihood)")
    plt.legend()
    plt.title(title)
    plt.savefig(fname)


def fixed_tree_convergence(
    T: nx.DiGraph, raw_params, params, learning_rate, steps, gcn
):
    tape_graphs = {}
    root = [v for v in T.nodes if T.in_degree[v] == 0][0]
    populate_tape_graphs(T, root, tape_graphs, params)
    ensure_fixed_tape_graphs(tape_graphs, params)
    params, history = optimize_likelihood(
        T, raw_params, tape_graphs, params.m, params.dt, learning_rate, steps, gcn
    )

    return history, params


def tree_search_convergence(
    labels,
    leaves,
    params,
    learning_rate,
    inner_steps,
    last_steps,
    tolerance,
    gcn,
    max_steps,
    root_length,
    nni_edges,
    num_trees=5,
):
    T, params, ml, history = tree_search(
        labels,
        leaves,
        params,
        learning_rate,
        inner_steps,
        last_steps,
        gcn,
        tolerance,
        root_length,
        n=num_trees,
        max_steps=max_steps,
        nni_edges=nni_edges,
    )

    return T, params, ml, history


def compare_tree_distances(
    T,
    D,
    raw_params,
    params_,
    learning_rate,
    inner_steps,
    last_steps,
    tolerance,
    gcn,
    max_steps,
):
    """Compare the pairwise leaf distances in the true tree to the pairwise leaf distances in the reconstructed"""
    return 0


def config_best_tree(best_tree, edge_lengths, root_length):

    root = [v for v in best_tree.nodes if best_tree.in_degree[v] == 0][0]
    # update the edge lengths in best_tree
    edge_ordering = edge_order(best_tree)
    for i, (u, v) in enumerate(edge_ordering):
        best_tree[u][v]["weight"] = edge_lengths[i].item()

    best_tree.add_node(-1)
    best_tree.add_edge(-1, root, weight=root_length.item())

    return best_tree


if __name__ == "__main__":
    for seed in range(10, 20):
        print("Running on simulation:", seed)
        dt = 0.05
        m = 10
        root_length = 5.0

        file = f"./simulations/{seed}/sampled_leaves.csv"
        labels, leaves = load_tape_states(file)

        # initialize params
        params = ModelParams(
            0.1, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1.0, 1.0, m, dt
        )

        # tree search convergence
        print(params)

        if len(labels) > 25:
            nni_edges = 15
        else:
            nni_edges = None

        best_tree, best_params, ml, likelihood_history = tree_search_convergence(
            labels, leaves, params, 1e-2, 10, 500, 1e-10, 1.0, 10, root_length, nni_edges, 5
        )
        params, root_length, _ = constrained_model_params(best_params, m, dt)
        edge_lengths = constrained_branch_lengths(best_tree, best_params)

        params_to_json(params, f"./simulations/{seed}/reconstructed_params.json")

        best_tree = config_best_tree(best_tree, edge_lengths, root_length)
        print_tree(best_tree, f"./simulations/{seed}/reconstructed_tree.csv")

        plot_convergence_single(
            likelihood_history,
            "Tree Search Tree Convergence",
            f"./simulations/{seed}/tree_search_convergence.png",
        )
