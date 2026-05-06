/// This file is for another project -- disregard
use std::{
    fs::File,
    io::{self, BufWriter, Write},
};

use rand::Rng;
use rand_distr::{Distribution, Exp};

use crate::Tree;

#[derive(Clone)]
struct Taxa {
    sequence: u128,
    len: u128,
}

impl Taxa {
    fn edit_on_branch<R: Rng + ?Sized>(&self, len: f64, edit_rate: f64, rng: &mut R) -> Self {
        let mut updated_self = self.clone();
        let exp_dist = Exp::new(edit_rate).unwrap();
        let mut t = 0.;
        while t < len {
            let edit_t = exp_dist.sample(rng);

            if t + edit_t > len {
                break;
            } else {
                let i = rng.gen_range(0..self.len);
                updated_self.sequence ^= 1u128 << i;
                t += edit_t
            }
        }

        updated_self
    }
}

impl Tree<Taxa> {
    fn simulate_editing_with_rng<R: Rng + ?Sized>(
        &mut self,
        root: usize,
        root_len: f64,
        root_tape: Taxa,
        edit_r: f64,
        rng: &mut R,
    ) {
        let updated_root = root_tape.edit_on_branch(root_len, edit_r, rng);
        self.labeling.insert(root, updated_root.clone());

        let successors = self.successors.get(&root).unwrap();
        match successors[..] {
            [(c1, c1_len), (c2, c2_len)] => {
                self.simulate_editing_with_rng(c1, c1_len, updated_root.clone(), edit_r, rng);
                self.simulate_editing_with_rng(c2, c2_len, updated_root.clone(), edit_r, rng);
            }
            [] => {}
            _ => unreachable!("Tree must be binary"),
        }
    }

    fn write_leaf_tsv<W: Write>(&self, mut out: W, n_chars: usize) -> io::Result<()> {
        assert!(n_chars <= 128, "u128 only stores 128 bits");

        // Header
        write!(out, "taxon")?;
        for i in 0..n_chars {
            write!(out, "\tchar_{}", i + 1)?;
        }
        writeln!(out)?;

        // Leaf rows
        for leaf in self.leaves() {
            let taxa = self
                .labeling
                .get(&leaf)
                .unwrap_or_else(|| panic!("leaf {} has no label", leaf));

            write!(out, "Species_{}", leaf)?;

            // Print bits left-to-right as char_1 ... char_n
            // This uses the low bits of the u128:
            // char_1 = bit 0, char_2 = bit 1, ...
            for i in 0..n_chars {
                let bit = (taxa.sequence >> i) & 1;
                write!(out, "\t{}", bit)?;
            }
            writeln!(out)?;
        }

        Ok(())
    }

    fn write_leaves_to_file(&self, fname: String, n_chars: usize) {
        let file = File::create(fname).unwrap();
        let writer = BufWriter::new(file);
        self.write_leaf_tsv(writer, n_chars).unwrap();
    }
}

#[test]
fn binary_sequences() {
    for n in [5, 20, 50, 100, 128] {
        for size in [8., 10., 13., 15., 20.] {
            let mut rng = rand::thread_rng();
            let (mut tree, root_len) = Tree::<Taxa>::branching_process_seeded(3.0, 1.0, size, 999);
            let leaf_count = tree.leaf_count();
            let root_tape = Taxa {
                sequence: rng.r#gen(),
                len: n,
            };
            tree.simulate_editing_with_rng(0, root_len, root_tape, 1.5, &mut rng);
            tree.write_leaves_to_file(
                format!("./sims/{leaf_count}Taxa_{n}Characters.tsv"),
                n as usize,
            );
        }
    }
}
