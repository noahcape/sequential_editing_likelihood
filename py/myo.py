import json
from dataclasses import asdict
import csv
import networkx as nx
from typing import List
from model import TapeState, ModelParams
import pandas as pd


def print_tree(T, path: str):
    with open(path, "w") as f:
        f.write("source,target,weight\n")
        nx.write_edgelist(T, path, ",")
    return T

"""Write params to json"""
def params_to_json(params: ModelParams, path: str):
    params_dict = asdict(params)
    params_dict["eta"] = params_dict["eta"].tolist() 
    
    with open(path, "w") as f:
        json.dump(params_dict, f)
    

# leaf,tape,target,non-target
# 131,0,6;5;2,6;5
def print_leaf_labels(T, path: str):
    leaves = [v for v in T.nodes if T.out_degree(v) == 0]
    with open(path, newline="") as f:
        f.write("leaf,tape,target,non-target\n")
        for l in leaves:
            label = T.nodes[l]["label"]
            tape_str = TapeState.as_string(label)
            f.write(f"{l},0,{tape_str}\n")


def parse_tape(s: str) -> List[int]:
    return [int(x) for x in s.split(";") if x != ""]


def load_tape_states(path: str) -> list[TapeState]:
    states = []
    leaves = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            leaf = row["leaf"]
            leaves.append(int(leaf))

            leading = parse_tape(row["target"])
            lagging = parse_tape(row["non-target"])

            state = TapeState.new(leading, lagging)
            states.append(state)

    return states, leaves


def read_full_tree(path: str) -> nx.DiGraph:
    df = pd.read_csv(path)
    return nx.from_pandas_edgelist(
        df, "parent", "child", edge_attr=True, create_using=nx.DiGraph()
    )


def skeleton_tree(T: nx.DiGraph, root, leaves):
    leaves = set(leaves)
    sT = nx.DiGraph()

    def rec(u):
        """
        Returns:
            (kept_node, distance_to_kept_node)
        or:
            (None, 0) if this subtree contains no selected leaves
        """

        # leaf
        if T.out_degree(u) == 0:
            if u in leaves:
                sT.add_node(u)
                return u, 0
            return None, 0

        children = list(T.successors(u))
        assert len(children) == 2, "Tree must be binary"

        kept = []

        for v in children:
            node, dist = rec(v)
            if node is not None:
                w = T[u][v].get("weight", 1)
                kept.append((node, dist + w))

        # no selected leaves below u
        if len(kept) == 0:
            return None, 0

        # only one selected lineage below u:
        # suppress u and pass the lineage upward
        if len(kept) == 1:
            return kept[0]

        # two selected lineages below u:
        # u is part of the skeleton
        sT.add_node(u)
        for node, dist in kept:
            sT.add_edge(u, node, weight=dist)

        return u, 0

    kept_root, _ = rec(root)

    return sT


if __name__ == "__main__":
    T = nx.DiGraph()

    T.add_nodes_from(range(11))
    T.add_edge(0, 1, weight=1.0)
    T.add_edge(0, 2, weight=1.0)
    T.add_edge(1, 3, weight=1.0)
    T.add_edge(1, 4, weight=1.0)
    T.add_edge(2, 5, weight=1.0)
    T.add_edge(2, 6, weight=1.0)
    T.add_edge(3, 7, weight=1.0)
    T.add_edge(3, 8, weight=1.0)
    T.add_edge(4, 9, weight=1.0)
    T.add_edge(4, 10, weight=1.0)

    # leaves = [5,6,7,8,9,10]
    leaves = [5, 8, 7, 10]
    sT = skeleton_tree(T, 0, leaves)

    tree_path = "./corrected_sims/0/full_edgelist.csv"
    leaves_path = "./corrected_sims/0/sampled_leaves.csv"
    T = read_full_tree(tree_path)
    _, leaves = load_tape_states(leaves_path)
    sT = skeleton_tree(T, 1, leaves)
