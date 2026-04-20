from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import networkx as nx
import optax


@dataclass(frozen=True)
class ModelParams:
    rho: float
    eta: jnp.ndarray
    tau: float
    lambd: float
    m: int
    dt: float


ArrayLikeTree = Mapping[str, Any]
LOG_ZERO = -1.0e30


@dataclass(frozen=True)
class TapeState:
    leading: Tuple[int, ...]
    lagging: Tuple[int, ...]
    orientation: int

    EVEN = 0
    ODD = 1

    @staticmethod
    def empty() -> "TapeState":
        return TapeState((), (), TapeState.EVEN)

    @staticmethod
    def new(leading: Sequence[int], lagging: Sequence[int]) -> "TapeState":
        orientation = TapeState.EVEN if len(leading) == len(lagging) else TapeState.ODD
        return TapeState(tuple(leading), tuple(lagging), orientation)

    def edit(self, edit_idx: int) -> "TapeState":
        return TapeState(self.leading + (edit_idx,), self.lagging, TapeState.ODD)

    def transfer(self) -> "TapeState":
        return TapeState.new(self.leading, self.lagging + (self.leading[-1],))

    def divide(self) -> Tuple["TapeState", "TapeState"]:
        return (
            TapeState.new(self.leading, self.leading),
            TapeState.new(self.lagging, self.lagging),
        )

    def lca(self, other: "TapeState") -> "TapeState":
        lead = []
        lag = []
        for a, b in zip(self.leading, other.leading):
            if a != b:
                break
            lead.append(a)
        for a, b in zip(self.lagging, other.lagging):
            if a != b:
                break
            lag.append(a)
        return TapeState.new(lead, lag)

    def predecessors(self, params: ModelParams) -> List["TapeState"]:
        preds: List[TapeState] = []
        leading = list(self.leading)
        lagging = list(self.lagging)
        orientation = self.orientation

        if orientation == TapeState.EVEN and len(self.leading) < params.m:
            for gamma in range(len(params.eta)):
                preds.append(
                    TapeState(tuple(leading + [gamma]), tuple(lagging), TapeState.ODD)
                )

        while leading:
            if orientation == TapeState.EVEN and len(lagging) > 0:
                preds.append(TapeState(tuple(leading), tuple(lagging), orientation))
                lagging.pop()
                orientation = TapeState.ODD
            else:
                leading.pop()
                for gamma in range(len(params.eta)):
                    preds.append(
                        TapeState(tuple(leading + [gamma]), tuple(lagging), orientation)
                    )
                orientation = TapeState.EVEN
                
        preds.append(TapeState.empty())
        return preds


@dataclass(frozen=True)
class TapeGraphArrays:
    target_tape: TapeState
    states: Tuple[TapeState, ...]
    state_to_idx: Mapping[TapeState, int]
    divide_targets_py: Tuple[Tuple[int, int], ...]
    orientation: jnp.ndarray
    edit_targets: jnp.ndarray
    transfer_targets: jnp.ndarray
    divide_targets: jnp.ndarray


def build_tape_graph(target: TapeState, params: ModelParams) -> TapeGraphArrays:
    """Build the predecessor-state graph for one target tape using padded arrays."""
    states = tuple(set(target.predecessors(params)))
    state_to_idx = {state: i for i, state in enumerate(states)}
    n_eta = len(params.eta)

    orientation = jnp.array([state.orientation for state in states], dtype=jnp.int32)

    edit_rows = []
    transfer_rows = []
    divide_rows = []
    for state in states:
        edit_rows.append(
            [state_to_idx.get(state.edit(edit_idx), -1) for edit_idx in range(n_eta)]
        )
        if state.orientation == TapeState.ODD and state.leading:
            transfer_rows.append(state_to_idx.get(state.transfer(), -1))
        else:
            transfer_rows.append(-1)
        left, right = state.divide()
        divide_rows.append([state_to_idx.get(left, -1), state_to_idx.get(right, -1)])

    return TapeGraphArrays(
        target_tape=target,
        states=states,
        state_to_idx=state_to_idx,
        divide_targets_py=tuple(tuple(row) for row in divide_rows),
        orientation=orientation,
        edit_targets=jnp.array(edit_rows, dtype=jnp.int32),
        transfer_targets=jnp.array(transfer_rows, dtype=jnp.int32),
        divide_targets=jnp.array(divide_rows, dtype=jnp.int32),
    )


def u_analytic(lambd: float, rho: float, t: jnp.ndarray) -> jnp.ndarray:
    """Closed-form solution for the auxiliary U(t) process used by the branch ODE."""
    a = rho / (1.0 - rho)
    return 1.0 / (1.0 + a * jnp.exp(t * lambd))


def gather_or_zero(values: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    """Gather by index while treating -1 sentinel entries as zero contribution."""
    clipped = jnp.clip(indices, 0)
    gathered = values[clipped]
    return jnp.where(indices >= 0, gathered, 0.0)


def d_ode(
    d: jnp.ndarray,
    graph: TapeGraphArrays,
    eta: jnp.ndarray,
    tau: float,
    lambd: float,
    u_at_t: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the likelihood ODE over all states in one tape graph."""
    h = jnp.sum(eta)

    divide_vals = gather_or_zero(d, graph.divide_targets)
    divide_sum = jnp.sum(divide_vals, axis=1)

    edit_vals = gather_or_zero(d, graph.edit_targets)
    edit_sum = jnp.sum(edit_vals * eta[None, :], axis=1)

    transfer_vals = gather_or_zero(d, graph.transfer_targets)

    # TODO: this is not correct should allow for division away from the targets though only into unsampled term
    even_rhs = -(lambd + h) * d + edit_sum + 0.5 * lambd * u_at_t * divide_sum
    odd_rhs = (
        -(lambd + tau) * d + tau * transfer_vals + 0.5 * lambd * u_at_t * divide_sum
    )
    return jnp.where(graph.orientation == TapeState.EVEN, even_rhs, odd_rhs)


def normalize_likelihood(d: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Rescale a likelihood vector to avoid underflow and return the log scale factor."""
    d = jnp.where(jnp.isfinite(d) & (d >= 0.0), d, 0.0)
    scale = jnp.max(d)
    safe_d = jnp.where(scale > 0.0, d / scale, d)
    log_scale = jnp.where(scale > 0.0, jnp.log(scale), 0.0)
    return safe_d, log_scale


def integrate_branch(
    d0: jnp.ndarray,
    graph: TapeGraphArrays,
    params: ModelParams,
    branch_length: jnp.ndarray | float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate one branch with Diffrax and return the normalized endpoint state."""
    eta = jnp.asarray(params.eta, dtype=d0.dtype)
    branch_length = jnp.asarray(branch_length, dtype=d0.dtype)
    dt0 = jnp.maximum(
        jnp.minimum(jnp.asarray(params.dt, dtype=d0.dtype), branch_length), 1e-6
    )

    def vf(t, y, _args):
        return d_ode(
            y,
            graph,
            eta,
            params.tau,
            params.lambd,
            u_analytic(params.lambd, params.rho, t),
        )

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(vf),
        solver=diffrax.Midpoint(),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        t0=0.0,
        t1=branch_length,
        dt0=dt0,
        y0=d0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-5),
        max_steps=100_000,
    )
    return normalize_likelihood(solution.ys[0])  # type: ignore


def leaf_likelihood(
    leaf_tape: TapeState, branch_length: jnp.ndarray | float, params: ModelParams
) -> Tuple[TapeGraphArrays, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Initialize and integrate the likelihood vector for a labeled leaf."""
    graph = build_tape_graph(leaf_tape, params)
    d0 = jnp.zeros((len(graph.states),), dtype=jnp.float32)
    d0 = d0.at[graph.state_to_idx[leaf_tape]].set(1.0)
    return graph, integrate_branch(d0, graph, params, branch_length)


def log_value(x: jnp.ndarray) -> jnp.ndarray:
    """Take a differentiable safe log with a small positive floor."""
    tiny = jnp.finfo(x.dtype).tiny
    return jnp.log(jnp.maximum(x, tiny))


def combine_children(
    left_graph: TapeGraphArrays,
    left_values: jnp.ndarray,
    left_log_scale: jnp.ndarray,
    right_graph: TapeGraphArrays,
    right_values: jnp.ndarray,
    right_log_scale: jnp.ndarray,
    params: ModelParams,
) -> Tuple[TapeGraphArrays, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Combine two child likelihood vectors into the parent pre-branch vector."""
    parent_tape = left_graph.target_tape.lca(right_graph.target_tape)
    parent_graph = build_tape_graph(parent_tape, params)

    def child_log(
        graph: TapeGraphArrays,
        values: jnp.ndarray,
        offset: jnp.ndarray,
        parent_idx: int,
    ):
        """Look up one parent state inside a child's graph and return its log-likelihood."""
        if parent_idx < 0:
            return jnp.asarray(LOG_ZERO)
        state = parent_graph.states[parent_idx]
        child_idx = graph.state_to_idx.get(state, -1)
        if child_idx < 0:
            return jnp.asarray(LOG_ZERO)
        return offset + log_value(values[child_idx])

    d_log = []
    branch_factor = jnp.log(0.5 * params.lambd)
    for left_idx, right_idx in parent_graph.divide_targets_py:

        term1 = child_log(
            left_graph, left_values, left_log_scale, left_idx
        ) + child_log(right_graph, right_values, right_log_scale, right_idx)
        term2 = child_log(
            left_graph, left_values, left_log_scale, right_idx
        ) + child_log(right_graph, right_values, right_log_scale, left_idx)
        d_log.append(
            jax.scipy.special.logsumexp(jnp.array([term1, term2])) + branch_factor
        )

    d_log = jnp.array(d_log)
    d, max_log = normalize_log_values(d_log)
    return parent_graph, (d, max_log)


def normalize_log_values(log_values: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Normalize log-values while handling the all-impossible case."""
    max_log = jnp.max(log_values)
    is_valid = jnp.isfinite(max_log)
    safe_max_log = jnp.where(is_valid, max_log, 0.0)
    values = jnp.where(
        is_valid & jnp.isfinite(log_values),
        jnp.exp(log_values - safe_max_log),
        0.0,
    )
    log_scale = jnp.where(is_valid, max_log, 0.0)
    return values, log_scale


def propagate_likelihood(
    graph: TapeGraphArrays,
    values: jnp.ndarray,
    log_scale: jnp.ndarray,
    branch_length: jnp.ndarray | float,
    params: ModelParams,
) -> Tuple[TapeGraphArrays, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Integrate a normalized likelihood vector along one branch and accumulate scale."""
    branch_values, branch_log_scale = integrate_branch(
        values, graph, params, branch_length
    )
    return graph, (branch_values, log_scale + branch_log_scale)


def root_initial_frequencies(
    graph: TapeGraphArrays, root_length: jnp.ndarray | float, params: ModelParams
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate the root-start distribution from the empty tape along the root branch."""
    empty_idx = graph.state_to_idx[TapeState.empty()]
    d0 = jnp.zeros((len(graph.states),), dtype=jnp.float32)
    d0 = d0.at[empty_idx].set(1.0)
    return integrate_branch(d0, graph, params, root_length)


def combine_with_root_frequencies(
    tree_graph: TapeGraphArrays,
    tree_values: jnp.ndarray,
    tree_log_scale: jnp.ndarray,
    root_graph: TapeGraphArrays,
    root_values: jnp.ndarray,
    root_log_scale: jnp.ndarray,
) -> jnp.ndarray | float:
    """Combine the tree likelihood vector with the root-start distribution in log-space."""
    root_log_sum = -jnp.inf
    for idx, state in enumerate(tree_graph.states):
        root_idx = root_graph.state_to_idx.get(state, -1)
        if root_idx < 0:
            continue
        term = (
            tree_log_scale
            + log_value(tree_values[idx])
            + root_log_scale
            + log_value(root_values[root_idx])
        )
        root_log_sum = jax.scipy.special.logsumexp(jnp.array([root_log_sum, term]))
    return root_log_sum


def rec_log_likelihood(
    tree: nx.DiGraph,
    node,
    branch_length: jnp.ndarray | float,
    params: ModelParams,
    branch_lengths: Mapping[Tuple[Any, Any], jnp.ndarray | float] | None = None,
) -> Tuple[TapeGraphArrays, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Recursively compute the log-likelihood on a binary tree"""
    if tree.out_degree(node) == 0:
        # node is a leaf
        return leaf_likelihood(tree.nodes[node]["label"], branch_length, params)
    else:
        # interior node
        children = list(tree.successors(node))
        assert len(children) == 2
        left, right = children
        if branch_lengths is None:
            left_length = tree.get_edge_data(node, left)["weight"]
            right_length = tree.get_edge_data(node, right)["weight"]
        else:
            left_length = branch_lengths[(node, left)]
            right_length = branch_lengths[(node, right)]
        (left_graph, (left_vals, left_scales)) = rec_log_likelihood(
            tree, left, left_length, params, branch_lengths
        )
        (right_graph, (right_vals, right_scales)) = rec_log_likelihood(
            tree, right, right_length, params, branch_lengths
        )
        parent_graph, (parent_values, parent_scale) = combine_children(
            left_graph,
            left_vals,
            left_scales,
            right_graph,
            right_vals,
            right_scales,
            params,
        )
        return propagate_likelihood(
            parent_graph, parent_values, parent_scale, branch_length, params
        )


def log_likelihood(
    tree: nx.DiGraph,
    params: ModelParams,
    root_length: jnp.ndarray | float,
    branch_lengths: Mapping[Tuple[Any, Any], jnp.ndarray | float] | None = None,
) -> jnp.ndarray | float:
    """
    Compute the log-likelihood of a fixed tree.
    1a. compute the tape graph from the set of leaf states
    1. compute the log-likelihood up the tree
    2. combine children at each interval vertex
    3. combine at the end with the root frequencies using root_length
    """

    root = [v for v in tree.nodes if tree.in_degree(v) == 0]
    assert len(root) == 1
    root = root[0]
    tree_graph, (tree_values, tree_log_scale) = rec_log_likelihood(
        tree, root, root_length, params, branch_lengths
    )
    root_graph = build_tape_graph(tree_graph.target_tape, params)
    root_values, root_log_scale = root_initial_frequencies(
        root_graph, root_length, params
    )
    return combine_with_root_frequencies(
        tree_graph,
        tree_values,
        tree_log_scale,
        root_graph,
        root_values,
        root_log_scale,
    )


def edge_order(tree: nx.DiGraph) -> Tuple[Tuple[Any, Any], ...]:
    """Return a stable edge order for packing branch lengths into a JAX array."""
    return tuple(tree.edges())


def branch_length_map(
    tree: nx.DiGraph, branch_lengths: jnp.ndarray
) -> Mapping[Tuple[Any, Any], jnp.ndarray]:
    """Build an edge-to-length mapping from a flat branch-length vector."""
    edges = edge_order(tree)
    assert branch_lengths.shape[0] == len(edges)
    return {edge: branch_lengths[i] for i, edge in enumerate(edges)}


def positive_transform(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Map unconstrained reals to strictly positive values."""
    return jax.nn.softplus(x) + eps


def unit_interval_transform(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Map unconstrained reals to the open unit interval."""
    return jax.nn.sigmoid(x) * (1.0 - 2.0 * eps) + eps


def positive_inverse_transform(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Approximate inverse of positive_transform for initializing raw parameters."""
    shifted = jnp.maximum(jnp.asarray(x) - eps, eps)
    return jnp.log(jnp.expm1(shifted))


def unit_interval_inverse_transform(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Approximate inverse of unit_interval_transform for initializing raw parameters."""
    x = jnp.asarray(x)
    scaled = (x - eps) / (1.0 - 2.0 * eps)
    scaled = jnp.clip(scaled, eps, 1.0 - eps)
    return jnp.log(scaled) - jnp.log1p(-scaled)


def constrained_model_params(
    raw_params: ArrayLikeTree, m: int, dt: float
) -> Tuple[ModelParams, jnp.ndarray, jnp.ndarray]:
    """Convert unconstrained optimization variables into model parameters and lengths."""
    rho = unit_interval_transform(jnp.asarray(raw_params["rho"], dtype=jnp.float32))
    eta = positive_transform(jnp.asarray(raw_params["eta"], dtype=jnp.float32))
    tau = positive_transform(jnp.asarray(raw_params["tau"], dtype=jnp.float32))
    lambd = positive_transform(jnp.asarray(raw_params["lambd"], dtype=jnp.float32))
    root_length = positive_transform(
        jnp.asarray(raw_params["root_length"], dtype=jnp.float32)
    )
    branch_lengths = positive_transform(
        jnp.asarray(raw_params["branch_lengths"], dtype=jnp.float32)
    )

    params = ModelParams(
        rho=rho,  # type: ignore
        eta=eta,
        tau=tau,  # type: ignore
        lambd=lambd,  # type: ignore
        m=m,
        dt=dt,
    )
    return params, root_length, branch_lengths


def likelihood_from_raw_params(
    tree: nx.DiGraph, raw_params: ArrayLikeTree, m: int, dt: float
) -> jnp.ndarray | float:
    """Evaluate the tree log-likelihood from unconstrained trainable variables."""
    params, root_length, branch_lengths = constrained_model_params(raw_params, m, dt)
    return log_likelihood(
        tree,
        params,
        root_length,
        branch_lengths=branch_length_map(tree, branch_lengths),
    )


def neg_log_likelihood_from_raw_params(
    tree: nx.DiGraph, raw_params: ArrayLikeTree, m: int, dt: float
) -> jnp.ndarray | float:
    """Convenience loss for optimizers that minimize objectives."""
    logl = likelihood_from_raw_params(tree, raw_params, m, dt)
    return -logl


def init_raw_params(
    tree: nx.DiGraph,
    eta: Sequence[float],
    rho: float,
    tau: float,
    lambd: float,
    root_length: float,
) -> dict[str, jnp.ndarray]:
    """Pack an initial guess into unconstrained variables for optimization."""
    branch_lengths = jnp.array(
        [tree.get_edge_data(u, v)["weight"] for u, v in edge_order(tree)],
        dtype=jnp.float32,
    )
    return {
        "rho": unit_interval_inverse_transform(jnp.asarray(rho, dtype=jnp.float32)),
        "eta": positive_inverse_transform(jnp.asarray(eta, dtype=jnp.float32)),
        "tau": positive_inverse_transform(jnp.asarray(tau, dtype=jnp.float32)),
        "lambd": positive_inverse_transform(jnp.asarray(lambd, dtype=jnp.float32)),
        "root_length": positive_inverse_transform(
            jnp.asarray(root_length, dtype=jnp.float32)
        ),
        "branch_lengths": positive_inverse_transform(branch_lengths),
    }


def optimize_likelihood(
    tree: nx.DiGraph,
    raw_params: ArrayLikeTree,
    m: int,
    dt: float,
    learning_rate: float = 1e-2,
    steps: int = 100,
    grad_clip_norm: float = 1.0,
) -> tuple[ArrayLikeTree, list[float]]:
    """Optimize the negative log-likelihood with Optax over model and branch parameters."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(raw_params)

    loss_fn = lambda params: neg_log_likelihood_from_raw_params(tree, params, m, dt)

    @eqx.filter_jit
    def step_fn(params, state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=1e3, neginf=-1e3), grads
        )
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, loss

    history: list[float] = []
    params = raw_params
    state = opt_state
    for _ in range(steps):
        params, state, loss = step_fn(params, state)
        # print("LOSS:", loss)
        history.append(float(loss))

    return params, history  # type: ignore


def summarize_gradient_tree(grads: ArrayLikeTree) -> dict[str, dict[str, float | bool]]:
    """Summarize gradient health for each top-level parameter block."""
    summary: dict[str, dict[str, float | bool]] = {}
    for key, value in grads.items():
        arr = jnp.asarray(value)
        finite = jnp.isfinite(arr)
        summary[key] = {
            "all_finite": bool(jnp.all(finite)),
            "any_nan": bool(jnp.any(jnp.isnan(arr))),
            "any_inf": bool(jnp.any(jnp.isinf(arr))),
            "max_abs": float(jnp.max(jnp.abs(jnp.nan_to_num(arr, nan=0.0)))),
            "mean_abs": float(jnp.mean(jnp.abs(jnp.nan_to_num(arr, nan=0.0)))),
        }
    return summary


def diagnose_gradients(
    tree: nx.DiGraph,
    raw_params: ArrayLikeTree,
    m: int,
    dt: float,
) -> tuple[float, ArrayLikeTree, dict[str, dict[str, float | bool]]]:
    """Compute one loss/gradient evaluation and summarize gradient pathologies."""
    loss_fn = lambda params: neg_log_likelihood_from_raw_params(tree, params, m, dt)
    loss, grads = jax.value_and_grad(loss_fn)(raw_params)
    summary = summarize_gradient_tree(grads)
    return float(loss), grads, summary


def _finite_scalar_summary(x: jnp.ndarray | float) -> dict[str, float | bool]:
    """Summarize one scalar objective value."""
    arr = jnp.asarray(x)
    return {
        "value": float(arr),
        "is_finite": bool(jnp.isfinite(arr)),
        "is_nan": bool(jnp.isnan(arr)),
        "is_inf": bool(jnp.isinf(arr)),
    }


def diagnose_objective_gradients(
    objective,
    raw_params: ArrayLikeTree,
) -> dict[str, Any]:
    """Run value_and_grad on an objective and summarize the resulting gradients."""
    value, grads = jax.value_and_grad(objective)(raw_params)
    return {
        "objective": _finite_scalar_summary(value),
        "gradients": summarize_gradient_tree(grads),
    }


def gradient_ablation_diagnostics(
    tree: nx.DiGraph,
    raw_params: ArrayLikeTree,
    m: int,
    dt: float,
) -> dict[str, Any]:
    """Diagnose which likelihood block introduces bad gradients using stop-gradient cuts."""

    def full_loss(params):
        return neg_log_likelihood_from_raw_params(tree, params, m, dt)

    def root_only_loss(params):
        model_params, root_length, branch_lengths = constrained_model_params(
            params, m, dt
        )
        tree_graph, (tree_values, tree_log_scale) = rec_log_likelihood(
            tree,
            [v for v in tree.nodes if tree.in_degree(v) == 0][0],
            root_length,
            model_params,
            branch_length_map(tree, branch_lengths),
        )
        tree_values = jax.lax.stop_gradient(tree_values)
        tree_log_scale = jax.lax.stop_gradient(tree_log_scale)
        root_graph = build_tape_graph(tree_graph.target_tape, model_params)
        root_values, root_log_scale = root_initial_frequencies(
            root_graph, root_length, model_params
        )
        return -combine_with_root_frequencies(
            tree_graph,
            tree_values,
            tree_log_scale,
            root_graph,
            root_values,
            root_log_scale,
        )

    def tree_only_loss(params):
        model_params, root_length, branch_lengths = constrained_model_params(
            params, m, dt
        )
        tree_graph, (tree_values, tree_log_scale) = rec_log_likelihood(
            tree,
            [v for v in tree.nodes if tree.in_degree(v) == 0][0],
            root_length,
            model_params,
            branch_length_map(tree, branch_lengths),
        )
        root_graph = build_tape_graph(tree_graph.target_tape, model_params)
        root_values, root_log_scale = root_initial_frequencies(
            root_graph, root_length, model_params
        )
        root_values = jax.lax.stop_gradient(root_values)
        root_log_scale = jax.lax.stop_gradient(root_log_scale)
        return -combine_with_root_frequencies(
            tree_graph,
            tree_values,
            tree_log_scale,
            root_graph,
            root_values,
            root_log_scale,
        )

    def branch_solve_only_loss(params):
        model_params, root_length, branch_lengths = constrained_model_params(
            params, m, dt
        )
        leaf = next(v for v in tree.nodes if tree.out_degree(v) == 0)
        label = tree.nodes[leaf]["label"]
        graph = build_tape_graph(label, model_params)
        d0 = jnp.zeros((len(graph.states),), dtype=jnp.float32)
        d0 = d0.at[graph.state_to_idx[label]].set(1.0)
        values, log_scale = integrate_branch(
            d0,
            graph,
            model_params,
            branch_lengths[0] if branch_lengths.shape[0] else root_length,
        )
        return -(jnp.sum(values) + log_scale)

    diagnostics = {
        "full": diagnose_objective_gradients(full_loss, raw_params),
        "root_only_tree_stopped": diagnose_objective_gradients(
            root_only_loss, raw_params
        ),
        "tree_only_root_stopped": diagnose_objective_gradients(
            tree_only_loss, raw_params
        ),
        "single_branch_solve": diagnose_objective_gradients(
            branch_solve_only_loss, raw_params
        ),
    }
    return diagnostics


def build_test_tree() -> nx.DiGraph:
    edges = [
        (0, 1, 1.2),
        (0, 2, 0.8),
        (1, 3, 0.2),
        (1, 4, 2.4),
        (2, 5, 0.9),
        (2, 6, 1.1),
    ]
    labeling = [
        (3, TapeState((0, 1), (0,), 1)),
        (4, TapeState((0, 1, 1), (0, 1), 1)),
        (5, TapeState((1, 2), (1,), 1)),
        (6, TapeState((1, 2), (1, 2), 0)),
    ]
    tree = nx.DiGraph()
    for v, label in labeling:
        tree.add_node(v, label=label)

    for u, v, w in edges:
        tree.add_edge(u, v, weight=w)

    return tree


def test_autodif():
    tree = build_test_tree()
    raw = init_raw_params(
        tree,
        eta=[0.1, 0.1, 0.1],
        rho=0.1,
        tau=0.2,
        lambd=1.0,
        root_length=1.5,
    )

    raw_opt, history = optimize_likelihood(
        tree,
        raw,
        m=3,
        dt=0.05,
        learning_rate=1e-2,
        steps=1000,
    )

    print(constrained_model_params(raw_opt, 3, 0.05))


def find_grad_issue():
    tree = build_test_tree()
    raw = init_raw_params(
        tree,
        eta=[1.4, 1.8, 2.2],
        rho=0.1,
        tau=0.2,
        lambd=1.0,
        root_length=1.5,
    )

    diag = gradient_ablation_diagnostics(tree, raw, m=3, dt=0.01)
    for name, result in diag.items():
        print(name)
        print(result["objective"])
        print(result["gradients"])


def test_state_graph():
    tape = TapeState((1, 2), (1,), 1)
    params = ModelParams(
        rho=0.1,
        eta=jnp.ones(3, dtype=float),
        tau=0.2,
        lambd=1.5,
        m=3,
        dt=0.5,
    )
    graph = build_tape_graph(tape, params)
    print(params.eta)
    print("divide targets", graph.divide_targets)
    print("transfer targets", graph.transfer_targets)
    print("edit targets", graph.edit_targets)
    print("orientation", graph.orientation)
    print("divide targets py", graph.divide_targets_py)
    print("state to idx")
    for s in graph.state_to_idx:
        print(s, graph.state_to_idx[s])
    print("states")
    for s in graph.states:
        print(s)
    print("target tape", graph.target_tape)


if __name__ == "__main__":
    # test_state_graph()

    test_autodif()
    # find_grad_issue()
