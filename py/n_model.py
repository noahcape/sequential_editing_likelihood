from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import jax
import jax.numpy as jnp
import networkx as nx

from model import (
    LOG_ZERO,
    MAX_STATES,
    ModelParams,
    TapeState,
    branch_length_map,
    constrained_branch_lengths,
    constrained_model_params,
    edge_order,
    ensure_fixed_tape_graphs,
    gather_or_zero,
    log_value,
    normalize_log_values,
    populate_tape_graphs,
    positive_transform,
    positive_inverse_transform,
    solve_branch_arrays,
    unit_interval_inverse_transform,
    unit_interval_transform,
)


def _flatten_model_params(params: ModelParams):
    children = (
        jnp.asarray(params.rho),
        jnp.asarray(params.eta),
        jnp.asarray(params.tau),
        jnp.asarray(params.lambd),
        jnp.asarray(params.dt),
    )
    return children, params.m


def _unflatten_model_params(m: int, children):
    rho, eta, tau, lambd, dt = children
    return ModelParams(rho=rho, eta=eta, tau=tau, lambd=lambd, m=m, dt=dt)


jax.tree_util.register_pytree_node(
    ModelParams, _flatten_model_params, _unflatten_model_params
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EncodedTree:
    """Fixed-shape array representation of one rooted binary tree."""

    postorder_nodes: Tuple[Any, ...]
    left_child: jnp.ndarray
    right_child: jnp.ndarray
    parent: jnp.ndarray
    is_leaf: jnp.ndarray
    branch_lengths: jnp.ndarray
    leaf_state_idx: jnp.ndarray
    active_mask: jnp.ndarray
    orientation: jnp.ndarray
    edit_targets: jnp.ndarray
    transfer_targets: jnp.ndarray
    divide_targets: jnp.ndarray
    root_pos: int
    empty_idx: int

    def tree_flatten(self):
        children = (
            self.left_child,
            self.right_child,
            self.parent,
            self.is_leaf,
            self.branch_lengths,
            self.leaf_state_idx,
            self.active_mask,
            self.orientation,
            self.edit_targets,
            self.transfer_targets,
            self.divide_targets,
            jnp.asarray(self.root_pos, dtype=jnp.int32),
            jnp.asarray(self.empty_idx, dtype=jnp.int32),
        )
        aux_data = len(self.postorder_nodes)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            left_child,
            right_child,
            parent,
            is_leaf,
            branch_lengths,
            leaf_state_idx,
            active_mask,
            orientation,
            edit_targets,
            transfer_targets,
            divide_targets,
            root_pos,
            empty_idx,
        ) = children
        return cls(
            postorder_nodes=tuple(range(aux_data)),
            left_child=left_child,
            right_child=right_child,
            parent=parent,
            is_leaf=is_leaf,
            branch_lengths=branch_lengths,
            leaf_state_idx=leaf_state_idx,
            active_mask=active_mask,
            orientation=orientation,
            edit_targets=edit_targets,
            transfer_targets=transfer_targets,
            divide_targets=divide_targets,
            root_pos=root_pos,
            empty_idx=empty_idx,
        )


def encode_tree(
    tree: nx.DiGraph,
    params: ModelParams,
    branch_lengths: Mapping[Tuple[Any, Any], jnp.ndarray | float] | None = None,
) -> EncodedTree:
    """Encode a NetworkX tree as fixed-shape arrays for JAX likelihood evaluation."""
    root = [v for v in tree.nodes if tree.in_degree(v) == 0]
    assert len(root) == 1
    root = root[0]

    tape_graphs = {}
    target_by_node = {}

    def target_tape(node):
        if tree.out_degree(node) == 0:
            tape = tree.nodes[node]["label"]
        else:
            children = list(tree.successors(node))
            assert len(children) == 2
            tape = target_tape(children[0]).lca(target_tape(children[1]))
        target_by_node[node] = tape
        return tape

    target_tape(root)
    populate_tape_graphs(tree, root, tape_graphs, params)
    ensure_fixed_tape_graphs(tape_graphs, params)

    postorder_nodes = tuple(reversed(tuple(nx.topological_sort(tree))))
    node_to_pos = {node: i for i, node in enumerate(postorder_nodes)}
    n_nodes = len(postorder_nodes)

    left_child = [-1] * n_nodes
    right_child = [-1] * n_nodes
    parent = [-1] * n_nodes
    is_leaf = [False] * n_nodes
    node_branch_lengths = [0.0] * n_nodes
    leaf_state_idx = [0] * n_nodes
    active_masks = []
    orientations = []
    edit_targets = []
    transfer_targets = []
    divide_targets = []

    for pos, node in enumerate(postorder_nodes):
        children = list(tree.successors(node))
        is_leaf[pos] = len(children) == 0
        if children:
            assert len(children) == 2
            left_child[pos] = node_to_pos[children[0]]
            right_child[pos] = node_to_pos[children[1]]
        else:
            label = tree.nodes[node]["label"]
            graph = tape_graphs[target_by_node[node]]
            leaf_state_idx[pos] = graph.state_to_idx[label]

        preds = list(tree.predecessors(node))
        if preds:
            edge = (preds[0], node)
            parent[pos] = node_to_pos[preds[0]]
            if branch_lengths is None:
                node_branch_lengths[pos] = tree.get_edge_data(*edge)["weight"]
            else:
                node_branch_lengths[pos] = branch_lengths[edge]

        graph = tape_graphs[target_by_node[node]]
        active_masks.append(graph.active_mask)
        orientations.append(graph.orientation)
        edit_targets.append(graph.edit_targets)
        transfer_targets.append(graph.transfer_targets)
        divide_targets.append(graph.divide_targets)

    root_pos = node_to_pos[root]
    root_graph = tape_graphs[target_by_node[root]]
    return EncodedTree(
        postorder_nodes=postorder_nodes,
        left_child=jnp.asarray(left_child, dtype=jnp.int32),
        right_child=jnp.asarray(right_child, dtype=jnp.int32),
        parent=jnp.asarray(parent, dtype=jnp.int32),
        is_leaf=jnp.asarray(is_leaf, dtype=bool),
        branch_lengths=jnp.asarray(node_branch_lengths, dtype=jnp.float32),
        leaf_state_idx=jnp.asarray(leaf_state_idx, dtype=jnp.int32),
        active_mask=jnp.stack(active_masks),
        orientation=jnp.stack(orientations),
        edit_targets=jnp.stack(edit_targets),
        transfer_targets=jnp.stack(transfer_targets),
        divide_targets=jnp.stack(divide_targets),
        root_pos=root_pos,
        empty_idx=root_graph.state_to_idx[TapeState.empty()],
    )


def _integrate_node(
    values: jnp.ndarray,
    active_mask: jnp.ndarray,
    orientation: jnp.ndarray,
    edit_targets: jnp.ndarray,
    transfer_targets: jnp.ndarray,
    divide_targets: jnp.ndarray,
    params: ModelParams,
    branch_length: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return solve_branch_arrays(
        values,
        active_mask,
        orientation,
        edit_targets,
        transfer_targets,
        divide_targets,
        jnp.asarray(params.eta, dtype=values.dtype),
        jnp.asarray(params.tau, dtype=values.dtype),
        jnp.asarray(params.lambd, dtype=values.dtype),
        jnp.asarray(params.rho, dtype=values.dtype),
        branch_length,
        jnp.asarray(params.dt, dtype=values.dtype),
    )


def _child_log(
    child_values: jnp.ndarray,
    child_scale: jnp.ndarray,
    child_active: jnp.ndarray,
    parent_active_by_state: jnp.ndarray,
    indices: jnp.ndarray,
) -> jnp.ndarray:
    clipped = jnp.clip(indices, 0)
    vals = child_values[clipped]
    valid = (
        (indices >= 0)
        & parent_active_by_state[clipped]
        & child_active[clipped]
        & (vals > 0.0)
    )
    return jnp.where(valid, child_scale + log_value(vals), LOG_ZERO)


def _combine_children_arrays(
    parent_active: jnp.ndarray,
    parent_divide_targets: jnp.ndarray,
    left_active: jnp.ndarray,
    left_values: jnp.ndarray,
    left_scale: jnp.ndarray,
    right_active: jnp.ndarray,
    right_values: jnp.ndarray,
    right_scale: jnp.ndarray,
    lambd: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    left_idx = parent_divide_targets[:, 0]
    right_idx = parent_divide_targets[:, 1]
    term1 = _child_log(
        left_values, left_scale, left_active, parent_active, left_idx
    ) + _child_log(right_values, right_scale, right_active, parent_active, right_idx)
    term2 = _child_log(
        left_values, left_scale, left_active, parent_active, right_idx
    ) + _child_log(right_values, right_scale, right_active, parent_active, left_idx)
    d_log = jax.scipy.special.logsumexp(jnp.stack([term1, term2]), axis=0) + jnp.log(
        0.5 * lambd
    )
    d_log = jnp.where(parent_active, d_log, LOG_ZERO)
    return normalize_log_values(d_log)


def _root_rate_matrix_arrays(
    active_mask: jnp.ndarray,
    orientation: jnp.ndarray,
    edit_targets: jnp.ndarray,
    transfer_targets: jnp.ndarray,
    eta: jnp.ndarray,
    tau: jnp.ndarray,
) -> jnp.ndarray:
    q = jnp.zeros((MAX_STATES, MAX_STATES), dtype=eta.dtype)
    rows = jnp.arange(MAX_STATES, dtype=jnp.int32)

    edit_valid = (
        active_mask[:, None]
        & (orientation[:, None] == TapeState.EVEN)
        & (edit_targets >= 0)
    )
    edit_cols = jnp.clip(edit_targets, 0)
    edit_rates = jnp.where(edit_valid, eta[None, :], 0.0)
    q = q.at[rows[:, None], edit_cols].add(edit_rates)

    transfer_valid = active_mask & (transfer_targets >= 0)
    transfer_cols = jnp.clip(transfer_targets, 0)
    q = q.at[rows, transfer_cols].add(jnp.where(transfer_valid, tau, 0.0))

    q = q * active_mask[:, None] * active_mask[None, :]
    row_sums = jnp.sum(q, axis=1)
    return q.at[jnp.diag_indices(MAX_STATES)].set(-row_sums)


def _root_initial_frequencies_arrays(
    active_mask: jnp.ndarray,
    orientation: jnp.ndarray,
    edit_targets: jnp.ndarray,
    transfer_targets: jnp.ndarray,
    empty_idx: int,
    root_length: jnp.ndarray,
    params: ModelParams,
) -> jnp.ndarray:
    eta = jnp.asarray(params.eta, dtype=jnp.float32)
    tau = jnp.asarray(params.tau, dtype=jnp.float32)
    q = _root_rate_matrix_arrays(
        active_mask, orientation, edit_targets, transfer_targets, eta, tau
    )
    p = jax.scipy.linalg.expm(q * jnp.asarray(root_length, dtype=q.dtype))[empty_idx]
    p = jnp.where(active_mask, jnp.maximum(p, 0.0), 0.0)
    return p / jnp.maximum(jnp.sum(p), jnp.finfo(p.dtype).tiny)


def array_log_likelihood_with_branch_lengths(
    encoded: EncodedTree,
    params: ModelParams,
    root_length: jnp.ndarray | float,
    branch_lengths: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate likelihood using array topology data instead of NetworkX recursion."""
    n_nodes = encoded.left_child.shape[0]
    values = jnp.zeros((n_nodes, MAX_STATES), dtype=jnp.float32)
    scales = jnp.zeros((n_nodes,), dtype=jnp.float32)

    for pos in range(n_nodes):
        active = encoded.active_mask[pos]
        leaf_values = jnp.zeros((MAX_STATES,), dtype=jnp.float32).at[
            encoded.leaf_state_idx[pos]
        ].set(1.0)

        left_pos = jnp.clip(encoded.left_child[pos], 0)
        right_pos = jnp.clip(encoded.right_child[pos], 0)
        internal_values, internal_scale = _combine_children_arrays(
            active,
            encoded.divide_targets[pos],
            encoded.active_mask[left_pos],
            values[left_pos],
            scales[left_pos],
            encoded.active_mask[right_pos],
            values[right_pos],
            scales[right_pos],
            jnp.asarray(params.lambd, dtype=jnp.float32),
        )

        node_values = jnp.where(encoded.is_leaf[pos], leaf_values, internal_values)
        node_scale = jnp.where(encoded.is_leaf[pos], 0.0, internal_scale)
        branch_values, branch_scale = _integrate_node(
            node_values,
            active,
            encoded.orientation[pos],
            encoded.edit_targets[pos],
            encoded.transfer_targets[pos],
            encoded.divide_targets[pos],
            params,
            branch_lengths[pos],
        )
        values = values.at[pos].set(branch_values)
        scales = scales.at[pos].set(node_scale + branch_scale)

    root_pos = encoded.root_pos
    root_values = _root_initial_frequencies_arrays(
        encoded.active_mask[root_pos],
        encoded.orientation[root_pos],
        encoded.edit_targets[root_pos],
        encoded.transfer_targets[root_pos],
        encoded.empty_idx,
        jnp.asarray(root_length, dtype=jnp.float32),
        params,
    )
    tree_values = values[root_pos]
    possible = (tree_values > 0.0) & (root_values > 0.0) & encoded.active_mask[root_pos]
    terms = scales[root_pos] + log_value(tree_values) + log_value(root_values)
    terms = jnp.where(possible, terms, -jnp.inf)
    return jax.scipy.special.logsumexp(terms)


def array_log_likelihood(
    encoded: EncodedTree,
    params: ModelParams,
    root_length: jnp.ndarray | float,
) -> jnp.ndarray:
    """Evaluate likelihood using the branch lengths stored in the encoded tree."""
    return array_log_likelihood_with_branch_lengths(
        encoded, params, root_length, encoded.branch_lengths
    )


def array_ultrametric_branch_lengths(
    encoded: EncodedTree, height_increments_by_pos: jnp.ndarray
) -> jnp.ndarray:
    """Derive branch lengths from per-node height increments using array topology."""
    n_nodes = encoded.left_child.shape[0]
    heights = jnp.zeros((n_nodes,), dtype=height_increments_by_pos.dtype)

    for pos in range(n_nodes):
        left_pos = jnp.clip(encoded.left_child[pos], 0)
        right_pos = jnp.clip(encoded.right_child[pos], 0)
        child_height = jnp.maximum(heights[left_pos], heights[right_pos])
        height = jnp.where(
            encoded.is_leaf[pos],
            0.0,
            child_height + height_increments_by_pos[pos],
        )
        heights = heights.at[pos].set(height)

    parent_pos = jnp.clip(encoded.parent, 0)
    branch_lengths = heights[parent_pos] - heights
    return jnp.where(encoded.parent >= 0, branch_lengths, 0.0)


def initial_height_increments_by_pos(encoded: EncodedTree) -> jnp.ndarray:
    """Initialize per-node height increments from encoded branch lengths."""
    n_nodes = encoded.left_child.shape[0]
    heights: list[float] = [0.0] * n_nodes
    increments: list[float] = [1.0e-6] * n_nodes

    for pos in range(n_nodes):
        if bool(encoded.is_leaf[pos]):
            heights[pos] = 0.0
            continue

        left = int(encoded.left_child[pos])
        right = int(encoded.right_child[pos])
        child_heights = [heights[left], heights[right]]
        child_lengths = [
            float(encoded.branch_lengths[left]),
            float(encoded.branch_lengths[right]),
        ]
        node_height = max(
            child_heights[0] + child_lengths[0],
            child_heights[1] + child_lengths[1],
        )
        heights[pos] = node_height
        increments[pos] = max(node_height - max(child_heights), 1.0e-6)

    return jnp.asarray(increments, dtype=jnp.float32)


def init_array_raw_params(
    encoded: EncodedTree,
    eta,
    rho: float,
    tau: float,
    lambd: float,
    root_length: float,
) -> dict[str, jnp.ndarray]:
    """Pack initial parameters for the array optimizer."""
    return {
        "rho": unit_interval_inverse_transform(jnp.asarray(rho, dtype=jnp.float32)),
        "eta": positive_inverse_transform(jnp.asarray(eta, dtype=jnp.float32)),
        "tau": positive_inverse_transform(jnp.asarray(tau, dtype=jnp.float32)),
        "lambd": positive_inverse_transform(jnp.asarray(lambd, dtype=jnp.float32)),
        "root_length": positive_inverse_transform(
            jnp.asarray(root_length, dtype=jnp.float32)
        ),
        "height_increments_by_pos": positive_inverse_transform(
            initial_height_increments_by_pos(encoded)
        ),
    }


def constrain_array_raw_params(
    encoded: EncodedTree, raw_params: Mapping[str, Any], dt: float, m: int = 0
) -> tuple[ModelParams, jnp.ndarray, jnp.ndarray]:
    rho = unit_interval_transform(jnp.asarray(raw_params["rho"], dtype=jnp.float32))
    eta = positive_transform(jnp.asarray(raw_params["eta"], dtype=jnp.float32))
    tau = positive_transform(jnp.asarray(raw_params["tau"], dtype=jnp.float32))
    lambd = positive_transform(jnp.asarray(raw_params["lambd"], dtype=jnp.float32))
    root_length = positive_transform(
        jnp.asarray(raw_params["root_length"], dtype=jnp.float32)
    )
    height_increments = positive_transform(
        jnp.asarray(raw_params["height_increments_by_pos"], dtype=jnp.float32)
    )
    branch_lengths = array_ultrametric_branch_lengths(encoded, height_increments)
    params = ModelParams(rho=rho, eta=eta, tau=tau, lambd=lambd, m=m, dt=dt)
    return params, root_length, branch_lengths


def array_neg_log_likelihood_from_raw_params(
    encoded: EncodedTree, raw_params: Mapping[str, Any], dt: float
) -> jnp.ndarray:
    params, root_length, branch_lengths = constrain_array_raw_params(
        encoded, raw_params, dt
    )
    return -array_log_likelihood_with_branch_lengths(
        encoded, params, root_length, branch_lengths
    )


def _tree_l2_norm(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def _adam_init_like(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return zeros, zeros, jnp.asarray(0, dtype=jnp.int32)


@jax.jit
def array_adam_step(
    encoded: EncodedTree,
    raw_params: Mapping[str, Any],
    m_state: Mapping[str, Any],
    v_state: Mapping[str, Any],
    t: jnp.ndarray,
    learning_rate: jnp.ndarray,
    grad_clip_norm: jnp.ndarray,
    dt: jnp.ndarray,
):
    """One reusable jitted Adam step for array-encoded topologies."""
    loss, grads = jax.value_and_grad(
        array_neg_log_likelihood_from_raw_params, argnums=1
    )(
        encoded, raw_params, dt
    )
    grads = jax.tree_util.tree_map(
        lambda g: jnp.nan_to_num(g, nan=0.0, posinf=1.0e3, neginf=-1.0e3), grads
    )
    grad_norm = _tree_l2_norm(grads)
    scale = jnp.minimum(1.0, grad_clip_norm / jnp.maximum(grad_norm, 1.0e-12))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)

    t = t + 1
    beta1 = jnp.asarray(0.9, dtype=learning_rate.dtype)
    beta2 = jnp.asarray(0.999, dtype=learning_rate.dtype)
    eps = jnp.asarray(1.0e-8, dtype=learning_rate.dtype)
    m_state = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, m_state, grads)
    v_state = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * jnp.square(g), v_state, grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1.0 - beta1**t), m_state)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1.0 - beta2**t), v_state)
    raw_params = jax.tree_util.tree_map(
        lambda p, m, v: p - learning_rate * m / (jnp.sqrt(v) + eps),
        raw_params,
        m_hat,
        v_hat,
    )
    return raw_params, m_state, v_state, t, loss


def optimize_array_likelihood(
    encoded: EncodedTree,
    raw_params: Mapping[str, Any],
    dt: float,
    learning_rate: float = 1.0e-2,
    steps: int = 100,
    grad_clip_norm: float = 1.0,
) -> tuple[Mapping[str, Any], list[float]]:
    """Optimize an array-encoded tree with the reusable jitted Adam step."""
    m_state, v_state, t = _adam_init_like(raw_params)
    history: list[float] = []
    params = raw_params
    for _ in range(steps):
        params, m_state, v_state, t, loss = array_adam_step(
            encoded,
            params,
            m_state,
            v_state,
            t,
            jnp.asarray(learning_rate, dtype=jnp.float32),
            jnp.asarray(grad_clip_norm, dtype=jnp.float32),
            jnp.asarray(dt, dtype=jnp.float32),
        )
        history.append(float(loss))
    return params, history


def array_likelihood_from_raw_params(
    encoded: EncodedTree,
    raw_params: Mapping[str, Any],
    m: int,
    dt: float,
) -> jnp.ndarray:
    if "height_increments_by_pos" in raw_params:
        return -array_neg_log_likelihood_from_raw_params(encoded, raw_params, dt)

    params, root_length, _ = constrained_model_params(raw_params, m, dt)
    return array_log_likelihood(encoded, params, root_length)


def encode_tree_from_raw_params(
    tree: nx.DiGraph, raw_params: Mapping[str, Any], m: int, dt: float
) -> tuple[EncodedTree, ModelParams, jnp.ndarray]:
    params, root_length, _ = constrained_model_params(raw_params, m, dt)
    branch_lengths = constrained_branch_lengths(tree, raw_params)
    encoded = encode_tree(tree, params, branch_length_map(tree, branch_lengths))
    return encoded, params, root_length
