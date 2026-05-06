from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "fontconfig"))

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import networkx as nx

from myo import read_full_tree, skeleton_tree


Label = tuple[tuple[int, ...], tuple[int, ...]]


def parse_vector(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(";") if x != "")


def load_leaf_labels(path: str | Path) -> tuple[dict[str, Label], list[str]]:
    labels: dict[str, Label] = {}
    order: list[str] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            leaf = str(row["leaf"])
            labels[leaf] = (
                parse_vector(row.get("target", "")),
                parse_vector(row.get("non-target", "")),
            )
            order.append(leaf)

    return labels, order


def load_root_length(path: str | Path) -> float:
    with open(path) as f:
        params = json.load(f)
    return float(params.get("root_length", 0.0))


def root_node(tree: nx.DiGraph) -> Any:
    roots = [node for node in tree.nodes if tree.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}")
    return roots[0]


def load_tree(
    edgelist_path: str | Path,
    labels: dict[str, Label],
    *,
    prune_to_labeled_leaves: bool,
    drop_dummy_root: bool,
) -> nx.DiGraph:
    tree = read_full_tree(str(edgelist_path))

    if prune_to_labeled_leaves:
        root = root_node(tree)
        if drop_dummy_root:
            children = list(tree.successors(root))
            if len(children) != 1:
                raise ValueError("Expected dummy root to have exactly one child")
            root = children[0]
        tree = skeleton_tree(tree, root, list(labels.keys()))

    for leaf, label in labels.items():
        if leaf in tree:
            tree.nodes[leaf]["label"] = label

    return tree


def subtree_leaf_sets(tree: nx.DiGraph) -> dict[Any, frozenset[str]]:
    root = root_node(tree)
    leaves_by_node: dict[Any, frozenset[str]] = {}

    def rec(node: Any) -> frozenset[str]:
        children = list(tree.successors(node))
        if not children:
            leaves_by_node[node] = frozenset([str(node)])
            return leaves_by_node[node]
        leaves = frozenset().union(*(rec(child) for child in children))
        leaves_by_node[node] = leaves
        return leaves

    rec(root)
    return leaves_by_node


def topology_signature(leaves: frozenset[str]) -> tuple[int, tuple[str, ...]]:
    return (-len(leaves), tuple(sorted(leaves)))


def oriented_leaf_order(
    tree: nx.DiGraph, reference_order: list[str] | None = None
) -> tuple[dict[Any, list[Any]], list[str]]:
    """Choose child flips that keep this tree planar and close to a reference order."""
    root = root_node(tree)
    leaves_by_node = subtree_leaf_sets(tree)
    reference_rank = (
        {leaf: i for i, leaf in enumerate(reference_order)}
        if reference_order is not None
        else {}
    )
    ordered_by_node: dict[Any, list[Any]] = {}

    def child_key(child: Any) -> tuple[float, int, tuple[str, ...]]:
        leaves = leaves_by_node[child]
        ranks = [reference_rank[leaf] for leaf in leaves if leaf in reference_rank]
        if ranks:
            return (sum(ranks) / len(ranks), min(ranks), tuple(sorted(leaves)))
        size_key, label_key = topology_signature(leaves)
        return (float(size_key), len(reference_rank), label_key)

    def rec(node: Any) -> list[str]:
        children = list(tree.successors(node))
        if not children:
            return [str(node)]
        ordered_children = sorted(children, key=child_key)
        ordered_by_node[node] = ordered_children
        leaves: list[str] = []
        for child in ordered_children:
            leaves.extend(rec(child))
        return leaves

    return ordered_by_node, rec(root)


def comparison_leaf_orders(
    true_tree: nx.DiGraph, reconstructed_tree: nx.DiGraph, iterations: int = 6
) -> tuple[dict[Any, list[Any]], list[str], dict[Any, list[Any]], list[str]]:
    true_children, true_order = oriented_leaf_order(true_tree)
    reconstructed_children, reconstructed_order = oriented_leaf_order(reconstructed_tree)

    for _ in range(iterations):
        true_children, true_order = oriented_leaf_order(true_tree, reconstructed_order)
        reconstructed_children, reconstructed_order = oriented_leaf_order(
            reconstructed_tree, true_order
        )

    return true_children, true_order, reconstructed_children, reconstructed_order


def tree_layout(
    tree: nx.DiGraph,
    leaf_order: list[str],
    child_order: dict[Any, list[Any]],
    root_length: float = 0.0,
) -> tuple[dict[Any, float], dict[Any, float], list[Any]]:
    root = root_node(tree)
    leaves = [node for node in tree.nodes if tree.out_degree(node) == 0]
    leaf_rank = {leaf: i for i, leaf in enumerate(leaf_order)}
    leaves = sorted(leaves, key=lambda leaf: (leaf_rank[str(leaf)], str(leaf)))

    x: dict[Any, float] = {root: float(root_length)}

    def set_x(node: Any) -> None:
        for child in child_order.get(node, list(tree.successors(node))):
            x[child] = x[node] + float(tree[node][child].get("weight", 1.0))
            set_x(child)

    set_x(root)

    y: dict[Any, float] = {leaf: float(i) for i, leaf in enumerate(leaves)}

    def set_y(node: Any) -> float:
        if tree.out_degree(node) == 0:
            return y[node]
        child_ys = [
            set_y(child) for child in child_order.get(node, list(tree.successors(node)))
        ]
        y[node] = sum(child_ys) / len(child_ys)
        return y[node]

    set_y(root)
    return x, y, leaves


def integer_color_map(labels: dict[str, Label]) -> dict[int, Any]:
    values = sorted(
        {
            value
            for target, non_target in labels.values()
            for vector in (target, non_target)
            for value in vector
        }
    )
    cmap = plt.get_cmap("tab20")
    return {value: cmap(i % cmap.N) for i, value in enumerate(values)}


def draw_label_vector(
    ax: Axes,
    x0: float,
    y0: float,
    values: tuple[int, ...],
    colors: dict[int, Any],
    *,
    cell_width: float,
    cell_height: float,
    show_values: bool,
) -> None:
    for i, value in enumerate(values):
        rect = Rectangle(
            (x0 + i * cell_width, y0),
            cell_width,
            cell_height,
            facecolor=colors[value],
            edgecolor="white",
            linewidth=0.35,
            clip_on=False,
        )
        ax.add_patch(rect)
        if show_values and cell_width >= 0.11 and cell_height >= 0.12:
            ax.text(
                x0 + (i + 0.5) * cell_width,
                y0 + cell_height / 2,
                str(value),
                ha="center",
                va="center",
                fontsize=5,
                color="black",
                clip_on=False,
            )


def draw_tree(
    ax: Axes,
    tree: nx.DiGraph,
    labels: dict[str, Label],
    leaf_order: list[str],
    child_order: dict[Any, list[Any]],
    colors: dict[int, Any],
    *,
    title: str,
    root_length: float = 0.0,
    show_leaf_ids: bool,
    show_label_values: bool,
) -> None:
    root = root_node(tree)
    x, y, leaves = tree_layout(tree, leaf_order, child_order, root_length)
    max_x = max(x.values()) if x else 1.0
    x_pad = max(max_x * 0.035, 0.5)
    label_x = max_x + x_pad
    max_vector_len = max(
        [1]
        + [
            max(len(target), len(non_target))
            for target, non_target in labels.values()
        ]
    )
    cell_width = max(max_x * 0.018, 0.18)
    cell_height = 0.32

    if root_length > 0:
        ax.plot(
            [0.0, x[root]],
            [y[root], y[root]],
            color="black",
            linewidth=0.8,
        )

    for parent, child in tree.edges:
        ax.plot(
            [x[parent], x[child]],
            [y[child], y[child]],
            color="black",
            linewidth=0.8,
        )

    for node in tree.nodes:
        children = list(tree.successors(node))
        if children:
            child_ys = [y[child] for child in children]
            ax.plot(
                [x[node], x[node]],
                [min(child_ys), max(child_ys)],
                color="black",
                linewidth=0.8,
            )

    id_offset = max_x * 0.01
    for leaf in leaves:
        leaf_label = labels.get(str(leaf))
        if show_leaf_ids:
            ax.text(
                label_x - id_offset,
                y[leaf],
                str(leaf),
                ha="right",
                va="center",
                fontsize=6,
                color="#333333",
            )
        if leaf_label is None:
            continue
        target, non_target = leaf_label
        draw_label_vector(
            ax,
            label_x,
            y[leaf] - cell_height - 0.02,
            target,
            colors,
            cell_width=cell_width,
            cell_height=cell_height,
            show_values=show_label_values,
        )
        draw_label_vector(
            ax,
            label_x,
            y[leaf] + 0.02,
            non_target,
            colors,
            cell_width=cell_width,
            cell_height=cell_height,
            show_values=show_label_values,
        )

    ax.set_title(title, fontsize=12)
    ax.set_ylim(len(leaves) - 0.5, -0.5)
    ax.set_xlim(-x_pad, label_x + max_vector_len * cell_width + x_pad)
    ax.set_xlabel("branch length")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw horizontal phylogenies with colored two-vector leaf labels."
    )
    parser.add_argument(
        "--true-tree",
        default="simulated_data/0/full_edgelist.csv",
        help="CSV edge list for the true tree.",
    )
    parser.add_argument(
        "--true-params",
        default="simulated_data/0/asymmetric_params.json",
        help="JSON params file containing the true tree root_length.",
    )
    parser.add_argument(
        "--reconstructed-tree",
        default="simulated_data/0/reconstructed_tree.csv",
        help="CSV edge list for the reconstructed tree.",
    )
    parser.add_argument(
        "--labels",
        default="simulated_data/0/sampled_leaves.csv",
        help="Leaf label CSV with leaf,target,non-target columns.",
    )
    parser.add_argument(
        "--output",
        default="simulated_data/0/labeled_tree_comparison.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--single-tree",
        help="Draw only this edge-list CSV instead of a true/reconstructed comparison.",
    )
    parser.add_argument(
        "--single-title",
        default="Tree",
        help="Title used with --single-tree.",
    )
    parser.add_argument(
        "--no-true-prune",
        action="store_true",
        help="Draw --true-tree directly instead of pruning it to labeled leaves.",
    )
    parser.add_argument(
        "--true-has-no-dummy-root",
        action="store_true",
        help="Do not drop the true tree's dummy root before pruning.",
    )
    parser.add_argument(
        "--hide-leaf-ids",
        action="store_true",
        help="Hide leaf ids next to the colored labels.",
    )
    parser.add_argument(
        "--hide-label-values",
        action="store_true",
        help="Hide integer text inside label color cells.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=16.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        help="Figure height in inches. Defaults to scale with leaf count.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels, label_order = load_leaf_labels(args.labels)
    colors = integer_color_map(labels)
    leaf_count = max(len(label_order), 1)
    fig_height = args.fig_height or max(7.0, 0.18 * leaf_count)

    if args.single_tree:
        tree = load_tree(
            args.single_tree,
            labels,
            prune_to_labeled_leaves=False,
            drop_dummy_root=False,
        )
        child_order, leaf_order = oriented_leaf_order(tree)
        fig, axes = plt.subplots(1, 1, figsize=(args.fig_width, fig_height), dpi=args.dpi)
        draw_tree(
            axes,
            tree,
            labels,
            leaf_order,
            child_order,
            colors,
            title=args.single_title,
            show_leaf_ids=not args.hide_leaf_ids,
            show_label_values=not args.hide_label_values,
        )
    else:
        true_root_length = load_root_length(args.true_params)
        true_tree = load_tree(
            args.true_tree,
            labels,
            prune_to_labeled_leaves=not args.no_true_prune,
            drop_dummy_root=not args.true_has_no_dummy_root,
        )
        reconstructed_tree = load_tree(
            args.reconstructed_tree,
            labels,
            prune_to_labeled_leaves=False,
            drop_dummy_root=False,
        )
        (
            true_child_order,
            true_leaf_order,
            reconstructed_child_order,
            reconstructed_leaf_order,
        ) = comparison_leaf_orders(true_tree, reconstructed_tree)
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(args.fig_width, fig_height),
            dpi=args.dpi,
            constrained_layout=True,
        )
        draw_tree(
            axes[0],
            true_tree,
            labels,
            true_leaf_order,
            true_child_order,
            colors,
            title="True tree",
            root_length=true_root_length,
            show_leaf_ids=not args.hide_leaf_ids,
            show_label_values=not args.hide_label_values,
        )
        draw_tree(
            axes[1],
            reconstructed_tree,
            labels,
            reconstructed_leaf_order,
            reconstructed_child_order,
            colors,
            title="Reconstructed tree",
            show_leaf_ids=not args.hide_leaf_ids,
            show_label_values=not args.hide_label_values,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
