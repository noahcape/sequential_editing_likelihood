use std::time::Instant;

use rand::{SeedableRng as _, rngs::StdRng};
use sequential_editing_likelihood::{ModelParams, TapeState, Tree};

struct SimulationCase {
    name: &'static str,
    shape: f64,
    rate: f64,
    t_max: f64,
    min_leaves: usize,
    max_leaves: usize,
    hill_edges: usize,
    hill_iters: usize,
    seed: u64,
}

fn larger_test_params() -> ModelParams {
    ModelParams::new(0.1, &[1.4, 1.6, 1.8, 2.0, 2.2, 2.4], 0.35, 1.0, 6, 0.02)
}

fn build_simulated_tree(
    case: &SimulationCase,
    params: &ModelParams,
) -> (Tree<TapeState>, f64, u64) {
    let mut best_candidate: Option<(Tree<TapeState>, f64, u64, usize)> = None;

    for offset in 0..4096_u64 {
        let tree_seed = case.seed + offset;
        let edit_seed = case.seed ^ (offset.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let (mut tree, root_len) =
            Tree::branching_process_seeded(case.shape, case.rate, case.t_max, tree_seed);
        let leaves = tree.leaf_count();

        let distance = if leaves < case.min_leaves {
            case.min_leaves - leaves
        } else if leaves > case.max_leaves {
            leaves - case.max_leaves
        } else {
            0
        };

        let should_replace = best_candidate
            .as_ref()
            .map(|(_, _, _, best_distance)| distance < *best_distance)
            .unwrap_or(true);
        if should_replace {
            best_candidate = Some((tree.clone(), root_len, tree_seed, distance));
        }

        if leaves < case.min_leaves || leaves > case.max_leaves {
            continue;
        }

        tree.simulate_editing_seeded(0, root_len, TapeState::empty(), params, edit_seed);
        return (tree, root_len, tree_seed);
    }

    if let Some((mut tree, root_len, tree_seed, distance)) = best_candidate {
        let edit_seed = case.seed ^ (tree_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let leaves = tree.leaf_count();
        tree.simulate_editing_seeded(0, root_len, TapeState::empty(), params, edit_seed);
        println!(
            "warning: case={} used closest tree with {} leaves (target {}..={}, distance {})",
            case.name, leaves, case.min_leaves, case.max_leaves, distance
        );
        return (tree, root_len, tree_seed);
    }

    panic!(
        "failed to generate a tree for case '{}' in range {}..={}",
        case.name, case.min_leaves, case.max_leaves
    );
}

#[test]
fn simulated_trees_likelihood_timing() {
    let params = larger_test_params();
    let cases = [
        SimulationCase {
            name: "small",
            shape: 2.0,
            rate: 1.10,
            t_max: 5.0,
            min_leaves: 8,
            max_leaves: 20,
            hill_edges: 0,
            hill_iters: 10,
            seed: 11,
        },
        SimulationCase {
            name: "medium",
            shape: 2.0,
            rate: 0.75,
            t_max: 10.0,
            min_leaves: 24,
            max_leaves: 48,
            hill_edges: 6,
            hill_iters: 25,
            seed: 29,
        },
        SimulationCase {
            name: "large",
            shape: 2.0,
            rate: 0.50,
            t_max: 20.0,
            min_leaves: 64,
            max_leaves: 96,
            hill_edges: 6,
            hill_iters: 20,
            seed: 53,
        },
        SimulationCase {
            name: "xlarge",
            shape: 2.0,
            rate: 0.40,
            t_max: 30.0,
            min_leaves: 128,
            max_leaves: 176,
            hill_edges: 4,
            hill_iters: 15,
            seed: 97,
        },
        SimulationCase {
            name: "xxlarge",
            shape: 2.0,
            rate: 0.33,
            t_max: 40.0,
            min_leaves: 200,
            max_leaves: 260,
            hill_edges: 4,
            hill_iters: 10,
            seed: 151,
        },
        SimulationCase {
            name: "xxxlarge",
            shape: 2.0,
            rate: 0.33,
            t_max: 50.0,
            min_leaves: 300,
            max_leaves: 560,
            hill_edges: 1,
            hill_iters: 0,
            seed: 17,
        },
    ];

    println!(
        "params: rho={:.3} tau={:.3} lambda={:.3} m={} eta_len={}",
        params.rho,
        params.tau,
        params.lambda,
        params.m,
        params.eta.len()
    );

    for case in &cases {
        let (mut tree, root_len, used_seed) = build_simulated_tree(case, &params);
        let started = Instant::now();
        let log_likelihood = tree.log_likelihood(0, root_len, &params);
        let elapsed = started.elapsed();

        tree.permute_leaves(
            50.min(case.min_leaves),
            &mut StdRng::seed_from_u64(used_seed),
        );
        let permuted_log_likelihood = tree.log_likelihood(0, root_len, &params);
        let hill_started = Instant::now();
        let climbed = tree
            .hill_climbing(root_len, &params, case.hill_edges, case.hill_iters)
            .expect("hill climbing should return a tree");
        let hill_elapsed = hill_started.elapsed();
        let climbed_log_likelihood = climbed.log_likelihood(0, root_len, &params);

        println!(
            "case={} seed={} vertices={} leaves={} root_len={:.4} shape={:.2} rate={:.2} log_likelihood={:.4e} likelihood_ms={:.3}, permuted_log_likelihood={:.4e} hill_log_likelihood={:.8e} hill_ms={:.3}",
            case.name,
            used_seed,
            tree.vertex_count(),
            tree.leaf_count(),
            root_len,
            case.shape,
            case.rate,
            log_likelihood,
            elapsed.as_secs_f64() * 1_000.0,
            permuted_log_likelihood,
            climbed_log_likelihood,
            hill_elapsed.as_secs_f64() * 1_000.0
        );

        if !log_likelihood.is_finite() || !climbed_log_likelihood.is_finite() {
            println!(
                "Lilkelihood is -inf! but... Cherry with empty LCA: {:?}",
                tree.cherry_lcas().iter().any(|t| t.len() < 1)
            );
        } else {
            println!(
                "Cherry with empty LCA: {:?}",
                tree.cherry_lcas().iter().any(|t| t.len() < 1)
            );
        }

        assert!(
            climbed_log_likelihood >= permuted_log_likelihood - 1e-10,
            "hill climbing should not decrease log-likelihood"
        );
    }
}
