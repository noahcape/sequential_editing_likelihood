pub mod ode;
pub mod opt;
pub mod temp;
pub mod binary_characters;

use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Exp, Gamma, WeightedIndex};
use std::collections::{HashMap, HashSet};

use crate::ode::{Rk4Scratch, rk4_d_ode};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TapeOrientation {
    Even,
    Odd,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct TapeState {
    leading: Vec<usize>,
    lagging: Vec<usize>,
    orientation: TapeOrientation,
}

impl TapeState {
    pub fn empty() -> Self {
        Self::new(vec![], vec![])
    }

    pub fn len(&self) -> usize {
        self.leading.len()
    }

    pub fn new(leading: Vec<usize>, lagging: Vec<usize>) -> Self {
        let orientation = match leading.len() == lagging.len() {
            true => TapeOrientation::Even,
            false => TapeOrientation::Odd,
        };

        Self {
            leading,
            lagging,
            orientation,
        }
    }

    pub fn sample_edit(&mut self, t: f64, params: &ModelParams) -> usize {
        let p_eta: Vec<f64> = params
            .eta
            .iter()
            .map(|rate| 1. - (-rate * t).exp())
            .collect();
        let dist = WeightedIndex::new(&p_eta).unwrap();
        let mut rng = thread_rng();

        let e = dist.sample(&mut rng);
        self.leading.push(e);
        self.orientation = TapeOrientation::Odd;
        e
    }

    fn sample_edit_with_rng<R: Rng + ?Sized>(
        &mut self,
        t: f64,
        params: &ModelParams,
        rng: &mut R,
    ) -> usize {
        let p_eta: Vec<f64> = params
            .eta
            .iter()
            .map(|rate| 1. - (-rate * t).exp())
            .collect();
        let dist = WeightedIndex::new(&p_eta).unwrap();

        let e = dist.sample(rng);
        self.leading.push(e);
        self.orientation = TapeOrientation::Odd;
        e
    }

    pub fn edit_transfer_on_branch(&self, length: f64, params: &ModelParams) -> Self {
        let mut rng = thread_rng();
        self.edit_transfer_on_branch_with_rng(length, params, &mut rng)
    }

    fn edit_transfer_on_branch_with_rng<R: Rng + ?Sized>(
        &self,
        length: f64,
        params: &ModelParams,
        rng: &mut R,
    ) -> Self {
        let mut updated_state = self.clone();
        let mut time = 0.0;

        while time < length && updated_state.leading.len() < params.m {
            let (edit, edit_wait): (usize, f64) = params
                .eta
                .iter()
                .map(|eta| Exp::new(*eta).unwrap().sample(rng))
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap();
            if time + edit_wait > length {
                break;
            }
            time += edit_wait;

            // Perform edit on target
            updated_state = updated_state.edit(edit);

            let remaining = length - time;
            let transfer_wait = Exp::new(params.tau).unwrap().sample(rng);

            if transfer_wait <= remaining {
                time += transfer_wait;
                updated_state = updated_state.transfer();
            } else {
                break;
            }
        }

        updated_state
    }

    pub fn all_predecessors(
        tapes: &[Self],
        params: &ModelParams,
    ) -> (HashMap<usize, TapeState>, HashMap<TapeState, usize>) {
        let states = tapes
            .iter()
            .flat_map(|s| s.predecessors(&params))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        let map = states
            .into_iter()
            .enumerate()
            .collect::<HashMap<usize, TapeState>>();

        let i_map = map
            .iter()
            .map(|(idx, state)| (state.clone(), *idx))
            .collect::<HashMap<TapeState, usize>>();

        (map, i_map)
    }

    pub fn edit(&self, edit: usize) -> Self {
        let mut leading = self.leading.clone();
        leading.push(edit);
        Self {
            leading,
            lagging: self.lagging.clone(),
            orientation: TapeOrientation::Odd,
        }
    }

    pub fn transfer(&self) -> Self {
        let mut lagging = self.lagging.clone();
        lagging.push(self.leading[self.leading.len() - 1]);
        Self::new(self.leading.clone(), lagging)
    }

    pub fn divide(&self) -> Vec<Self> {
        vec![
            Self {
                leading: self.leading.clone(),
                lagging: self.leading.clone(),
                orientation: TapeOrientation::Even,
            },
            Self {
                leading: self.lagging.clone(),
                lagging: self.lagging.clone(),
                orientation: TapeOrientation::Even,
            },
        ]
    }

    pub fn predecessors(&self, params: &ModelParams) -> Vec<TapeState> {
        let mut preds = Vec::new();
        preds.reserve(2 * self.leading.len());
        let mut leading = self.leading.to_vec();
        let mut lagging = self.lagging.to_vec();
        let mut orientation = self.orientation;

        if orientation == TapeOrientation::Even {
            if self.leading.len() < params.m {
                for gamma in 0..params.eta.len() {
                    let mut new_leading = leading.clone();
                    new_leading.push(gamma);
                    preds.push(TapeState {
                        leading: new_leading.clone(),
                        lagging: lagging.clone(),
                        orientation: TapeOrientation::Odd,
                    });
                }
            }
        }

        while !leading.is_empty() {
            match orientation {
                TapeOrientation::Even => {
                    preds.push(TapeState {
                        leading: leading.clone(),
                        lagging: lagging.clone(),
                        orientation,
                    });
                    lagging.pop();
                    orientation = TapeOrientation::Odd;
                }
                TapeOrientation::Odd => {
                    leading.pop();

                    for gamma in 0..params.eta.len() {
                        let mut new_leading = leading.clone();
                        new_leading.push(gamma);
                        preds.push(TapeState {
                            leading: new_leading.clone(),
                            lagging: lagging.clone(),
                            orientation,
                        });
                    }
                    orientation = TapeOrientation::Even;
                }
            }
        }

        preds.push(TapeState {
            leading: vec![],
            lagging: vec![],
            orientation: TapeOrientation::Even,
        });

        preds
    }

    /// Get the level that the Tape is in where m is the maximum length of the tape
    pub fn get_level(&self, m: usize) -> usize {
        match self.orientation {
            TapeOrientation::Even => 2 * (m - self.leading.len()),
            TapeOrientation::Odd => 2 * (m - self.leading.len()) + 1,
        }
    }

    fn lca(&self, other: &Self) -> Self {
        let mut leading = vec![];
        let mut lagging = vec![];

        for i in 0..(self.leading.len().min(other.leading.len())) {
            if self.leading[i] == other.leading[i] {
                leading.push(self.leading[i])
            } else {
                break;
            }
        }

        for i in 0..self.lagging.len().min(other.lagging.len()) {
            if self.lagging[i] == other.lagging[i] {
                lagging.push(self.lagging[i])
            } else {
                break;
            }
        }

        let orientation = if leading.len() == lagging.len() {
            TapeOrientation::Even
        } else {
            TapeOrientation::Odd
        };

        Self {
            leading,
            lagging,
            orientation,
        }
    }

    // compute the LCA of many tapes
    pub fn lca_many(tapes: &[&Self]) -> Self {
        if tapes.len() == 0 {
            TapeState {
                leading: vec![],
                lagging: vec![],
                orientation: TapeOrientation::Even,
            }
        } else {
            let (first, last) = tapes.split_at(1);
            let mut lca = first[0].clone();

            for next in last.iter() {
                lca = lca.lca(next);
            }

            lca
        }
    }
}

// TODO: take a closer look at the tape graph and what it is doing
pub(crate) struct TapeGraph {
    pub(crate) states: Vec<TapeState>,
    pub(crate) i_map: HashMap<TapeState, usize>,
    pub(crate) edit_targets: Vec<Vec<Option<usize>>>,
    pub(crate) transfer_targets: Vec<Option<usize>>,
    pub(crate) divide_targets: Vec<[Option<usize>; 2]>,
}

impl TapeGraph {
    fn new(target: &TapeState, params: &ModelParams) -> Self {
        let states = target
            .predecessors(params)
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        let i_map = states
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, state)| (state, idx))
            .collect::<HashMap<_, _>>();

        let mut edit_targets = Vec::with_capacity(states.len());
        let mut transfer_targets = Vec::with_capacity(states.len());
        let mut divide_targets = Vec::with_capacity(states.len());

        for state in &states {
            let edits = (0..params.eta.len())
                .map(|edit| i_map.get(&state.edit(edit)).copied())
                .collect::<Vec<_>>();
            let transfer = if state.orientation == TapeOrientation::Odd && !state.leading.is_empty()
            {
                i_map.get(&state.transfer()).copied()
            } else {
                None
            };
            let children = state.divide();
            let divide = [
                i_map.get(&children[0]).copied(),
                i_map.get(&children[1]).copied(),
            ];

            edit_targets.push(edits);
            transfer_targets.push(transfer);
            divide_targets.push(divide);
        }

        assert!(
            !states.is_empty(),
            "There should always be at least one possible predecessor"
        );

        Self {
            states,
            i_map,
            edit_targets,
            transfer_targets,
            divide_targets,
        }
    }
}

struct ScaledLikelihood {
    values: Vec<f64>,
    log_scale: f64,
    tape: TapeState,
}

fn log_value(value: f64) -> f64 {
    if value > 0.0 {
        value.ln()
    } else {
        f64::NEG_INFINITY
    }
}

fn logsumexp_pair(a: f64, b: f64) -> f64 {
    match (a.is_finite(), b.is_finite()) {
        (true, true) => {
            let m = a.max(b);
            m + ((a - m).exp() + (b - m).exp()).ln()
        }
        (true, false) => a,
        (false, true) => b,
        (false, false) => f64::NEG_INFINITY,
    }
}

fn normalize_log_values(log_values: &[f64]) -> (Vec<f64>, f64) {
    let max_log = log_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !max_log.is_finite() {
        return (vec![0.0; log_values.len()], 0.0);
    }

    let values = log_values
        .iter()
        .map(|&log_value| {
            if log_value.is_finite() {
                (log_value - max_log).exp()
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();

    (values, max_log)
}

fn normalize_likelihood(values: &mut [f64]) -> f64 {
    for value in values.iter_mut() {
        if !value.is_finite() || *value < 0.0 {
            *value = 0.0;
        }
    }

    let scale = values.iter().copied().fold(0.0_f64, f64::max);
    if scale > 0.0 {
        for value in values.iter_mut() {
            *value /= scale;
        }
        scale.ln()
    } else {
        0.0
    }
}

#[test]
fn test_predecessors() {
    let tape = TapeState {
        leading: vec![0, 1, 2],
        lagging: vec![0, 1],
        orientation: TapeOrientation::Odd,
    };

    // map from index to tape state
    let (map, _) = TapeState::all_predecessors(
        &tape.predecessors(&ModelParams::test()),
        &ModelParams::test(),
    );

    for (k, v) in map {
        println!("{} -> {:?}", k, v);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tree<N> {
    vertices: usize,
    successors: HashMap<usize, Vec<(usize, f64)>>,
    predecessors: HashMap<usize, (usize, f64)>,
    labeling: HashMap<usize, N>,
}

impl<N> Tree<N>
where
    N: Clone,
{
    pub fn new() -> Self {
        Tree {
            vertices: 0,
            successors: HashMap::new(),
            predecessors: HashMap::new(),
            labeling: HashMap::new(),
        }
    }

    pub fn add_root(&mut self, root: usize) -> usize {
        self.successors.insert(root, vec![]);
        self.vertices += 1;
        root
    }

    pub fn add_child(&mut self, parent: usize, child: usize, blen: f64) -> (usize, usize) {
        assert!(
            self.successors.contains_key(&parent) && self.successors.contains_key(&parent),
            "Parent must already be in the tree"
        );
        assert!(
            !self.successors.contains_key(&child) && !self.successors.contains_key(&child),
            "Child id cannot already be in the tree"
        );
        self.successors
            .get_mut(&parent)
            .unwrap()
            .push((child, blen));
        self.successors.insert(child, vec![]);
        self.predecessors.insert(child, (parent, blen));
        self.vertices += 1;

        (parent, child)
    }

    pub fn add_label(&mut self, node: usize, label: N) -> Option<N> {
        self.labeling.insert(node, label)
    }

    pub fn leaves(&self) -> Vec<&usize> {
        self.successors
            .iter()
            .filter_map(|(k, v)| if v.is_empty() { Some(k) } else { None })
            .collect::<Vec<_>>()
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices
    }

    pub fn leaf_count(&self) -> usize {
        self.leaves().len()
    }

    /// Assume that Self is ultra-metric
    pub fn height(&self, root: usize, b_len: f64) -> f64 {
        match self.successors.get(&root).unwrap()[..] {
            [] => b_len,
            [(c, n_b_len), (_, _)] => b_len + self.height(c, n_b_len),
            _ => panic!("Must be a binary tree"),
        }
    }

    // Simulate
    pub fn branching_process(shape: f64, rate: f64, t_max: f64) -> (Self, f64) {
        let mut rng = thread_rng();
        Self::branching_process_with_rng(shape, rate, t_max, &mut rng)
    }

    pub fn branching_process_seeded(shape: f64, rate: f64, t_max: f64, seed: u64) -> (Self, f64) {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::branching_process_with_rng(shape, rate, t_max, &mut rng)
    }

    fn branching_process_with_rng<R: Rng + ?Sized>(
        shape: f64,
        rate: f64,
        t_max: f64,
        rng: &mut R,
    ) -> (Self, f64) {
        let gamma = Gamma::new(shape, 1. / rate).unwrap();

        let mut tree = Tree::new();
        let root_len = gamma.sample(rng);
        tree.add_root(0);

        let mut idx = 1;
        let mut leaves = vec![(0, root_len)];
        loop {
            let mut new_leaves = vec![];
            let alive_leaves: Vec<_> = leaves.iter().filter(|(_, len)| *len < t_max).collect();
            if alive_leaves.is_empty() {
                break;
            }
            for &(leaf_idx, len) in alive_leaves {
                for bl in gamma.sample_iter(&mut *rng).take(2) {
                    let b_len = bl.min((t_max - len).max(0.0));
                    tree.add_child(leaf_idx, idx, b_len);
                    new_leaves.push((idx, len + b_len));
                    idx += 1;
                }
            }
            leaves = new_leaves;
        }

        (tree, root_len)
    }

    /// Permute some leaves
    pub fn permute_leaves<R: Rng + ?Sized>(&mut self, n: usize, rng: &mut R) {
        let leaves: Vec<usize> = self.leaves().into_iter().copied().collect();
        for _ in 0..n {
            let l1 = leaves[rng.gen_range(0..leaves.len())];
            let l2 = leaves[rng.gen_range(0..leaves.len())];

            if l1 == l2 {
                continue;
            }

            let v1 = self.labeling.remove(&l1).unwrap();
            let v2 = self.labeling.remove(&l2).unwrap();

            self.labeling.insert(l1, v2);
            self.labeling.insert(l2, v1);
        }
    }

    /// Preform an NNI move on the edge (u, v) producing two new trees
    pub fn nni(&self, u: usize, v: usize) -> Vec<Self> {
        let u_successors = self.successors.get(&u).unwrap();
        let v_successors = self.successors.get(&v).unwrap();
        assert!(u_successors.len() == 2, "v must be a binary internal node");
        assert!(v_successors.len() == 2, "u must be a binary internal node");
        assert!(
            u_successors.iter().any(|(c, _)| *c == v),
            "u and v must be connected by an edge."
        );

        let (a, a_len) = v_successors[0];
        let (b, b_len) = v_successors[1];
        let &(d, d_len) = u_successors
            .iter()
            .find(|(c, _)| *c != v)
            .expect("u has child other than v");
        let &(_, v_len) = u_successors
            .iter()
            .find(|(c, _)| *c == v)
            .expect("u has child v");

        let l_u = self.height(u, 0.);
        let l_v = l_u - v_len;
        let l_a = l_v - a_len;
        let l_b = l_v - b_len;
        let l_d = l_u - d_len;

        let mut a_tree = self.clone();
        a_tree.predecessors.insert(a, (u, l_u - l_a));
        a_tree.predecessors.insert(d, (v, l_v - l_d));
        a_tree
            .successors
            .insert(v, vec![(d, l_v - l_d), (b, l_v - l_b)]);
        a_tree
            .successors
            .insert(u, vec![(a, l_u - l_a), (v, v_len)]);

        let mut b_tree = self.clone();
        b_tree.predecessors.insert(b, (u, l_u - l_b));
        b_tree.predecessors.insert(d, (v, l_v - l_d));
        b_tree
            .successors
            .insert(v, vec![(d, l_v - l_d), (a, l_v - l_a)]);
        b_tree
            .successors
            .insert(u, vec![(b, l_u - l_b), (v, v_len)]);

        vec![a_tree, b_tree]
    }
}

impl Tree<TapeState> {
    fn possible_root_tape(&self) -> TapeState {
        let leaves: Vec<&TapeState> = self
            .leaves()
            .iter()
            .map(|l| self.labeling.get(l).unwrap())
            .collect();

        TapeState::lca_many(&leaves)
    }

    fn leaf_sibling_tapes(&self, leaf: &usize) -> TapeState {
        let (parent, _) = self.predecessors.get(&leaf).unwrap();
        TapeState::lca_many(
            &self
                .successors
                .get(parent)
                .unwrap()
                .iter()
                .map(|(i, _)| self.labeling.get(&i).unwrap())
                .collect::<Vec<_>>(),
        )
    }

    pub fn cherry_lcas(&self) -> HashSet<TapeState> {
        let mut lcas = HashSet::new();
        for l in self.leaves() {
            lcas.insert(self.leaf_sibling_tapes(l));
        }

        lcas
    }

    fn tape_graph<'a>(
        &self,
        tape: &TapeState,
        params: &ModelParams,
        cache: &'a mut HashMap<TapeState, TapeGraph>,
    ) -> &'a TapeGraph {
        cache
            .entry(tape.clone())
            .or_insert_with(|| TapeGraph::new(tape, params))
    }

    pub fn simulate_editing(
        &mut self,
        root: usize,
        root_len: f64,
        root_tape: TapeState,
        params: &ModelParams,
    ) -> TapeState {
        let mut rng = thread_rng();
        self.simulate_editing_with_rng(root, root_len, root_tape, params, &mut rng)
    }

    pub fn simulate_editing_seeded(
        &mut self,
        root: usize,
        root_len: f64,
        root_tape: TapeState,
        params: &ModelParams,
        seed: u64,
    ) -> TapeState {
        let mut rng = StdRng::seed_from_u64(seed);
        self.simulate_editing_with_rng(root, root_len, root_tape, params, &mut rng)
    }

    fn simulate_editing_with_rng<R: Rng + ?Sized>(
        &mut self,
        root: usize,
        root_len: f64,
        root_tape: TapeState,
        params: &ModelParams,
        rng: &mut R,
    ) -> TapeState {
        let updated_root = root_tape.edit_transfer_on_branch_with_rng(root_len, params, rng);
        self.labeling.insert(root, updated_root.clone());

        let successors = self.successors.get(&root).unwrap();
        match successors[..] {
            [(c1, c1_len), (c2, c2_len)] => {
                let divide = self.labeling.get(&root).unwrap().divide();
                let state1 =
                    self.simulate_editing_with_rng(c1, c1_len, divide[0].clone(), params, rng);
                let _ = self.simulate_editing_with_rng(c2, c2_len, divide[1].clone(), params, rng);

                state1
            }
            [] => updated_root,
            _ => unreachable!("Tree must be binary"),
        }
    }

    pub fn log_likelihood(&self, root: usize, root_len: f64, params: &ModelParams) -> f64 {
        let mut graph_cache = HashMap::new();
        let mut scratch = Rk4Scratch::default();

        // run forward time ODE to compute the initial probabilities
        let root_tape = self.possible_root_tape();
        self.tape_graph(&root_tape, params, &mut graph_cache);
        let mut init_d;
        let mut init_log_scale = 0.0;
        {
            let root_graph = graph_cache.get(&root_tape).unwrap();
            let empty_idx = root_graph
                .i_map
                .get(&TapeState {
                    leading: vec![],
                    lagging: vec![],
                    orientation: TapeOrientation::Even,
                })
                .unwrap();
            let mut t = 0.0;
            init_d = vec![0.; root_graph.states.len()];
            init_d[*empty_idx] = 1.;

            while t + params.dt <= root_len {
                rk4_d_ode(
                    &mut init_d,
                    &mut scratch,
                    params.dt,
                    params.lambda,
                    params.tau,
                    t,
                    params.rho,
                    &params.eta,
                    root_graph,
                );
                init_log_scale += normalize_likelihood(&mut init_d);
                t += params.dt;
            }
        }

        let scaled_tree =
            self.r_likelihood_cached(root, root_len, params, &mut graph_cache, &mut scratch);
        let root_i_map = &graph_cache.get(&root_tape).unwrap().i_map;
        let tree_graph = graph_cache.get(&scaled_tree.tape).unwrap();

        let mut root_log_sum = f64::NEG_INFINITY;
        for (idx, state) in tree_graph.states.iter().enumerate() {
            if let Some(root_idx) = root_i_map.get(state) {
                let term = scaled_tree.log_scale
                    + log_value(scaled_tree.values[idx])
                    + init_log_scale
                    + log_value(init_d[*root_idx]);
                root_log_sum = logsumexp_pair(root_log_sum, term);
            }
        }
        root_log_sum
    }

    pub fn likelihood(&self, root: usize, root_len: f64, params: &ModelParams) -> f64 {
        self.log_likelihood(root, root_len, params).exp()
    }

    /// Returns a vector of the probability of the tree arises given the root starts in that state
    /// Turn this into the likelhood by multiplying by the probability of the root starting in that state
    /// TODO: probability just the probabilty associated with the null state since we pass in the branch length of root
    pub fn r_likelihood(
        &self,
        root: usize,
        b_len: f64,
        params: &ModelParams,
    ) -> (
        Vec<f64>,
        HashMap<usize, TapeState>,
        HashMap<TapeState, usize>,
        TapeState,
    ) {
        let mut graph_cache = HashMap::new();
        let mut scratch = Rk4Scratch::default();
        let scaled = self.r_likelihood_cached(root, b_len, params, &mut graph_cache, &mut scratch);
        let graph = graph_cache.remove(&scaled.tape).unwrap();
        let map = graph
            .states
            .into_iter()
            .enumerate()
            .collect::<HashMap<usize, TapeState>>();
        let i_map = graph.i_map;
        let d = if scaled.log_scale.is_finite() {
            scaled
                .values
                .iter()
                .map(|value| value * scaled.log_scale.exp())
                .collect::<Vec<_>>()
        } else {
            vec![0.0; scaled.values.len()]
        };

        (d, map, i_map, scaled.tape)
    }

    fn r_likelihood_cached(
        &self,
        root: usize,
        b_len: f64,
        params: &ModelParams,
        graph_cache: &mut HashMap<TapeState, TapeGraph>,
        scratch: &mut Rk4Scratch,
    ) -> ScaledLikelihood {
        let successors = self.successors.get(&root).unwrap();
        match successors[..] {
            [] => {
                let tape = self.labeling.get(&root).unwrap();
                let graph = self.tape_graph(tape, params, graph_cache);
                let tape_idx = graph.i_map.get(tape).unwrap();
                let mut d = vec![0.; graph.states.len()];
                d[*tape_idx] = 1.;
                let mut log_scale = 0.0;

                let mut t = 0.0;
                while t + params.dt <= b_len {
                    rk4_d_ode(
                        &mut d,
                        scratch,
                        params.dt,
                        params.lambda,
                        params.tau,
                        t,
                        params.rho,
                        &params.eta,
                        graph,
                    );
                    log_scale += normalize_likelihood(&mut d);
                    t += params.dt;
                }

                ScaledLikelihood {
                    values: d,
                    log_scale,
                    tape: tape.clone(),
                }
            }
            [(c1, b_len_1), (c2, b_len_2)] => {
                let scaled_c1 = self.r_likelihood_cached(c1, b_len_1, params, graph_cache, scratch);
                let scaled_c2 = self.r_likelihood_cached(c2, b_len_2, params, graph_cache, scratch);

                let lca = scaled_c1.tape.lca(&scaled_c2.tape);
                self.tape_graph(&lca, params, graph_cache);
                let c1_graph = graph_cache.get(&scaled_c1.tape).unwrap();
                let c2_graph = graph_cache.get(&scaled_c2.tape).unwrap();
                let graph = graph_cache.get(&lca).unwrap();

                let mut d_log = vec![f64::NEG_INFINITY; graph.states.len()];
                for (i, children) in graph.divide_targets.iter().enumerate() {
                    let first_left = children[0]
                        .and_then(|idx| c1_graph.i_map.get(&graph.states[idx]).copied())
                        .map(|idx| scaled_c1.log_scale + log_value(scaled_c1.values[idx]))
                        .unwrap_or(f64::NEG_INFINITY);
                    let first_right = children[1]
                        .and_then(|idx| c2_graph.i_map.get(&graph.states[idx]).copied())
                        .map(|idx| scaled_c2.log_scale + log_value(scaled_c2.values[idx]))
                        .unwrap_or(f64::NEG_INFINITY);
                    let second_left = children[1]
                        .and_then(|idx| c1_graph.i_map.get(&graph.states[idx]).copied())
                        .map(|idx| scaled_c1.log_scale + log_value(scaled_c1.values[idx]))
                        .unwrap_or(f64::NEG_INFINITY);
                    let second_right = children[0]
                        .and_then(|idx| c2_graph.i_map.get(&graph.states[idx]).copied())
                        .map(|idx| scaled_c2.log_scale + log_value(scaled_c2.values[idx]))
                        .unwrap_or(f64::NEG_INFINITY);

                    let term1 = if first_left.is_finite() && first_right.is_finite() {
                        (0.5 * params.lambda).ln() + first_left + first_right
                    } else {
                        f64::NEG_INFINITY
                    };
                    let term2 = if second_left.is_finite() && second_right.is_finite() {
                        (0.5 * params.lambda).ln() + second_left + second_right
                    } else {
                        f64::NEG_INFINITY
                    };

                    d_log[i] = logsumexp_pair(term1, term2);
                }
                let (mut d, mut log_scale) = normalize_log_values(&d_log);

                let mut t = 0.0;
                while t + params.dt <= b_len {
                    rk4_d_ode(
                        &mut d,
                        scratch,
                        params.dt,
                        params.lambda,
                        params.tau,
                        t,
                        params.rho,
                        &params.eta,
                        graph,
                    );
                    log_scale += normalize_likelihood(&mut d);
                    t += params.dt;
                }

                ScaledLikelihood {
                    values: d,
                    log_scale,
                    tape: lca,
                }
            }
            _ => panic!("This should be a binary tree!"),
        }
    }
}

pub struct ModelParams {
    pub rho: f64,
    pub eta: Vec<f64>,
    pub tau: f64,
    pub lambda: f64,
    pub m: usize,
    pub dt: f64,
}

impl ModelParams {
    pub fn new(rho: f64, eta: &[f64], tau: f64, lambda: f64, m: usize, dt: f64) -> Self {
        ModelParams {
            rho,
            eta: eta.to_vec(),
            tau,
            lambda,
            m,
            dt,
        }
    }

    pub fn test() -> Self {
        let lambda = 1.0;
        let tau = 0.2;
        let eta = vec![2.0, 2.2, 1.8];
        let rho = 0.1;
        ModelParams {
            rho,
            eta,
            tau,
            lambda,
            m: 3,
            dt: 0.01,
        }
    }
}

#[test]
fn likelihood_bl_zero() {
    let mut tree: Tree<TapeState> = Tree {
        vertices: 0,
        successors: HashMap::new(),
        predecessors: HashMap::new(),
        labeling: HashMap::new(),
    };
    tree.add_root(0);
    tree.add_label(
        0,
        TapeState {
            leading: vec![],
            lagging: vec![],
            orientation: TapeOrientation::Even,
        },
    );

    let likelihood = tree.likelihood(0, 0., &ModelParams::test());
    // zero branch length is the sampling probability
    assert!(likelihood > 0.);
}

#[test]
fn likelihood_single_leaf() {
    let mut tree: Tree<TapeState> = Tree {
        vertices: 0,
        successors: HashMap::new(),
        predecessors: HashMap::new(),
        labeling: HashMap::new(),
    };
    tree.add_root(0);
    tree.add_label(
        0,
        TapeState {
            leading: vec![1],
            lagging: vec![1],
            orientation: TapeOrientation::Even,
        },
    );

    let l = tree.likelihood(0, 2., &ModelParams::test());
    assert!(l > 0.);
}

#[test]
fn likelihood_cherry() {
    let mut tree: Tree<TapeState> = Tree {
        vertices: 0,
        successors: HashMap::new(),
        predecessors: HashMap::new(),
        labeling: HashMap::new(),
    };
    tree.add_root(0);
    tree.add_child(0, 1, 5.0);
    tree.add_child(0, 2, 5.0);
    let mut tree_a = tree.clone();
    let mut tree_b = tree;
    tree_a.add_label(
        1,
        TapeState {
            leading: vec![1, 2],
            lagging: vec![1],
            orientation: TapeOrientation::Odd,
        },
    );
    tree_a.add_label(
        2,
        TapeState {
            leading: vec![1, 1],
            lagging: vec![1],
            orientation: TapeOrientation::Odd,
        },
    );

    tree_b.add_label(
        1,
        TapeState {
            leading: vec![1, 2],
            lagging: vec![1],
            orientation: TapeOrientation::Odd,
        },
    );
    tree_b.add_label(
        2,
        TapeState {
            leading: vec![2, 0],
            lagging: vec![2],
            orientation: TapeOrientation::Odd,
        },
    );

    let l_a = tree_a.likelihood(0, 5.0, &ModelParams::test());
    let l_b = tree_b.likelihood(0, 5.0, &ModelParams::test());

    assert!(l_a > l_b)
}

#[test]
fn better_likelihood_tree() {
    let mut tree_a: Tree<TapeState> = Tree {
        vertices: 0,
        successors: HashMap::new(),
        predecessors: HashMap::new(),
        labeling: HashMap::new(),
    };
    tree_a.add_root(0);
    tree_a.add_child(0, 1, 10.0);
    tree_a.add_child(0, 2, 5.0);
    tree_a.add_child(2, 3, 5.0);
    tree_a.add_child(2, 4, 5.0);
    let mut tree_b = tree_a.clone();

    let label_1 = TapeState {
        leading: vec![1, 2],
        lagging: vec![1],
        orientation: TapeOrientation::Odd,
    };
    let label_2 = TapeState {
        leading: vec![1, 0, 2],
        lagging: vec![1, 0],
        orientation: TapeOrientation::Odd,
    };
    let label_3 = TapeState {
        leading: vec![1, 0],
        lagging: vec![1, 0],
        orientation: TapeOrientation::Even,
    };
    // label tree_a
    tree_a.add_label(1, label_1.clone());
    tree_a.add_label(3, label_2.clone());
    tree_a.add_label(4, label_3.clone());

    // label tree_b
    tree_b.add_label(1, label_2);
    tree_b.add_label(3, label_1);
    tree_b.add_label(4, label_3);

    let l_a = tree_a.likelihood(0, 1.5, &ModelParams::test());
    let l_b = tree_b.likelihood(0, 1.5, &ModelParams::test());

    assert!(l_a > l_b);
}

#[test]
fn t_tree_length() {
    let mut tree: Tree<TapeState> = Tree {
        vertices: 0,
        successors: HashMap::new(),
        predecessors: HashMap::new(),
        labeling: HashMap::new(),
    };
    tree.add_root(0);
    tree.add_child(0, 1, 10.0);
    tree.add_child(0, 2, 5.0);
    tree.add_child(2, 3, 5.0);
    tree.add_child(2, 4, 5.0);

    assert_eq!(tree.height(0, 0.), 10.)
}

#[test]
fn test_simulate_editing() {
    let mut tree: Tree<TapeState> = Tree {
        vertices: 0,
        successors: HashMap::new(),
        predecessors: HashMap::new(),
        labeling: HashMap::new(),
    };
    tree.add_root(0);
    tree.add_child(0, 1, 10.0);
    tree.add_child(0, 2, 5.0);
    tree.add_child(2, 3, 5.0);
    tree.add_child(2, 4, 5.0);

    let _ = tree.simulate_editing(0, 1.5, TapeState::empty(), &ModelParams::test());
    println!("{:?}", tree.labeling);
}
