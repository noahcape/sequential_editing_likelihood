use crate::{ModelParams, TapeState, Tree};
use rand::seq::SliceRandom;
use rand::thread_rng;

impl Tree<TapeState> {
    fn internal_nni_edges(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();

        for (&child, &(parent, _)) in &self.predecessors {
            let parent_is_internal = self
                .successors
                .get(&parent)
                .map(|children| children.len() == 2)
                .unwrap_or(false);
            let child_is_internal = self
                .successors
                .get(&child)
                .map(|children| children.len() == 2)
                .unwrap_or(false);

            if parent_is_internal && child_is_internal {
                edges.push((parent, child));
            }
        }

        edges
    }

    fn candidate_nni_edges(&self, max_edges: usize) -> Vec<(usize, usize)> {
        let mut edges = self.internal_nni_edges();
        if max_edges == 0 || max_edges >= edges.len() {
            return edges;
        }

        let mut rng = thread_rng();
        edges.shuffle(&mut rng);
        edges.truncate(max_edges);
        edges
    }

    fn best_nni_neighbor(
        &self,
        root_len: f64,
        params: &ModelParams,
        max_edges: usize,
    ) -> Option<(Self, f64)> {
        let mut best_tree = None;
        let mut best_log_likelihood = f64::NEG_INFINITY;

        for (u, v) in self.candidate_nni_edges(max_edges) {
            for neighbor in self.nni(u, v) {
                let log_likelihood = neighbor.log_likelihood(0, root_len, params);
                if log_likelihood > best_log_likelihood {
                    best_log_likelihood = log_likelihood;
                    best_tree = Some(neighbor);
                }
            }
        }

        best_tree.map(|tree| (tree, best_log_likelihood))
    }

    pub fn hill_climbing(
        &self,
        root_len: f64,
        params: &ModelParams,
        n: usize,
        max_iters: usize,
    ) -> Option<Self> {
        let mut current = self.clone();
        let mut current_log_likelihood = current.log_likelihood(0, root_len, params);

        for _ in 0..max_iters {
            let Some((next_tree, next_log_likelihood)) =
                current.best_nni_neighbor(root_len, params, n)
            else {
                break;
            };

            if next_log_likelihood <= current_log_likelihood - 1e-10_f64.ln() {
                break;
            }

            current = next_tree;
            current_log_likelihood = next_log_likelihood;
        }

        Some(current)
    }
}

#[test]
fn test_hill_climbing() {
    use std::collections::HashMap;

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

    let label_1 = TapeState::new(vec![1, 2], vec![1]);
    let label_2 = TapeState::new(vec![1, 0, 2], vec![1, 0]);
    let label_3 = TapeState::new(vec![1, 0], vec![1, 0]);
    tree.add_label(1, label_2);
    tree.add_label(3, label_1);
    tree.add_label(4, label_3);

    let start_likelihood = tree.log_likelihood(0, 1.5, &ModelParams::test());
    let climbed = tree.hill_climbing(1.5, &ModelParams::test(), 0, 10).unwrap();
    let end_likelihood = climbed.log_likelihood(0, 1.5, &ModelParams::test());

    assert!(end_likelihood >= start_likelihood);
}
