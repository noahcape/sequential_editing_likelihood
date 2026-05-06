use std::{
    env,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{ModelParams, TapeState, Tree};

pub struct SimulationParams {
    pub output_dir: PathBuf,
    pub tree_seed: u64,
    pub edit_seed: u64,
    pub sample_seed: u64,
    pub tree_shape: f64,
    pub tree_rate: f64,
    pub t_max: f64,
    pub model: ModelParams,
}

impl Default for SimulationParams {
    fn default() -> Self {
        let seed = 0;
        let eta = vec![0.1; 10];
        Self {
            output_dir: PathBuf::from("simulations/0"),
            tree_seed: seed,
            edit_seed: seed ^ 0x9E37_79B9_7F4A_7C15,
            sample_seed: seed ^ 0xD1B5_4A32_D192_ED03,
            tree_shape: 2.0,
            tree_rate: 0.5,
            t_max: 20.0,
            model: ModelParams::new(0.1, &eta, 0.15, 1.0, 8, 0.05),
        }
    }
}

pub fn run_from_args() -> io::Result<()> {
    let params = parse_args(env::args().skip(1))?;
    run(params)
}

pub fn run(params: SimulationParams) -> io::Result<()> {
    fs::create_dir_all(&params.output_dir)?;

    let (mut tree, root_len) = Tree::<TapeState>::branching_process_seeded(
        params.tree_shape,
        params.tree_rate,
        params.t_max,
        params.tree_seed,
    );
    tree.simulate_editing_seeded(
        0,
        root_len,
        TapeState::empty(),
        &params.model,
        params.edit_seed,
    );

    let mut sample_rng = StdRng::seed_from_u64(params.sample_seed);
    let sampled_leaves = sample_leaves(&tree, params.model.rho, &mut sample_rng);

    write_edgelist(&tree, root_len, params.output_dir.join("full_edgelist.csv"))?;
    write_leaf_labels(
        &tree,
        tree.leaves().into_iter().copied(),
        params.output_dir.join("full_leaves.csv"),
    )?;
    write_leaf_labels(
        &tree,
        sampled_leaves.into_iter(),
        params.output_dir.join("sampled_leaves.csv"),
    )?;
    write_params(
        &params,
        root_len,
        tree.leaf_count(),
        params.output_dir.join("asymmetric_params.json"),
    )?;

    println!(
        "wrote simulation with {} leaves to {}",
        tree.leaf_count(),
        params.output_dir.display()
    );
    Ok(())
}

fn parse_args<I>(args: I) -> io::Result<SimulationParams>
where
    I: IntoIterator<Item = String>,
{
    let mut params = SimulationParams::default();
    let mut args = args.into_iter().collect::<Vec<_>>();
    if args.first().map(String::as_str) == Some("simulate") {
        args.remove(0);
    }

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--help" || flag == "-h" {
            print_usage();
            return Ok(params);
        }
        let value = args.get(i + 1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("missing value for argument {flag}"),
            )
        })?;

        match flag {
            "--out" => params.output_dir = PathBuf::from(value),
            "--seed" => {
                let seed = parse_value::<u64>(flag, value)?;
                params.tree_seed = seed;
                params.edit_seed = seed ^ 0x9E37_79B9_7F4A_7C15;
                params.sample_seed = seed ^ 0xD1B5_4A32_D192_ED03;
            }
            "--tree-seed" => params.tree_seed = parse_value(flag, value)?,
            "--edit-seed" => params.edit_seed = parse_value(flag, value)?,
            "--sample-seed" => params.sample_seed = parse_value(flag, value)?,
            "--shape" => params.tree_shape = parse_positive(flag, value)?,
            "--rate" | "--birth-rate" => params.tree_rate = parse_positive(flag, value)?,
            "--t-max" => params.t_max = parse_positive(flag, value)?,
            "--rho" => params.model.rho = parse_probability(flag, value)?,
            "--tau" => params.model.tau = parse_positive(flag, value)?,
            "--lambda" => params.model.lambda = parse_positive(flag, value)?,
            "--m" | "--max-tape-len" => params.model.m = parse_value(flag, value)?,
            "--dt" => params.model.dt = parse_positive(flag, value)?,
            "--eta" => params.model.eta = parse_eta(value)?,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument {flag}"),
                ));
            }
        }
        i += 2;
    }

    if params.model.eta.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--eta must contain at least one rate",
        ));
    }

    Ok(params)
}

fn parse_value<T>(flag: &str, value: &str) -> io::Result<T>
where
    T: std::str::FromStr,
{
    value.parse::<T>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid value for {flag}: {value}"),
        )
    })
}

fn parse_positive(flag: &str, value: &str) -> io::Result<f64> {
    let parsed = parse_value::<f64>(flag, value)?;
    if parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{flag} must be positive"),
        ))
    }
}

fn parse_probability(flag: &str, value: &str) -> io::Result<f64> {
    let parsed = parse_value::<f64>(flag, value)?;
    if (0.0..=1.0).contains(&parsed) {
        Ok(parsed)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{flag} must be between 0 and 1"),
        ))
    }
}

fn parse_eta(value: &str) -> io::Result<Vec<f64>> {
    value
        .split(',')
        .map(|part| {
            let rate = part.trim().parse::<f64>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid eta rate: {part}"),
                )
            })?;
            if rate > 0.0 {
                Ok(rate)
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "eta rates must be positive",
                ))
            }
        })
        .collect()
}

fn sample_leaves<R: Rng + ?Sized>(tree: &Tree<TapeState>, rho: f64, rng: &mut R) -> Vec<usize> {
    let mut leaves = tree.leaves().into_iter().copied().collect::<Vec<_>>();
    leaves.sort_unstable();
    leaves
        .into_iter()
        .filter(|_| rng.gen_bool(rho))
        .collect::<Vec<_>>()
}

fn write_edgelist<P: AsRef<Path>>(
    tree: &Tree<TapeState>,
    root_len: f64,
    path: P,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);
    writeln!(out, "parent,child,weight")?;
    writeln!(out, "0,1,{root_len}")?;

    let mut edges = tree
        .successors
        .iter()
        .flat_map(|(parent, children)| {
            children
                .iter()
                .map(move |(child, length)| (parent + 1, child + 1, *length))
        })
        .collect::<Vec<_>>();
    edges.sort_by_key(|(parent, child, _)| (*parent, *child));

    for (parent, child, length) in edges {
        writeln!(out, "{parent},{child},{length}")?;
    }

    Ok(())
}

fn write_leaf_labels<P, I>(tree: &Tree<TapeState>, leaves: I, path: P) -> io::Result<()>
where
    P: AsRef<Path>,
    I: IntoIterator<Item = usize>,
{
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);
    writeln!(out, "leaf,tape,target,non-target")?;

    let mut leaves = leaves.into_iter().collect::<Vec<_>>();
    leaves.sort_unstable();
    for leaf in leaves {
        let tape = tree.labeling.get(&leaf).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("leaf {leaf} has no simulated tape"),
            )
        })?;
        writeln!(
            out,
            "{},0,{},{}",
            leaf + 1,
            tape_part_to_string(&tape.leading),
            tape_part_to_string(&tape.lagging)
        )?;
    }

    Ok(())
}

fn tape_part_to_string(part: &[usize]) -> String {
    part.iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(";")
}

fn write_params<P: AsRef<Path>>(
    params: &SimulationParams,
    root_len: f64,
    leaf_count: usize,
    path: P,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);

    writeln!(out, "{{")?;
    writeln!(out, "  \"lambda\": {},", params.model.lambda)?;
    writeln!(out, "  \"tau\": {},", params.model.tau)?;
    writeln!(out, "  \"eta\": [")?;
    for (idx, eta) in params.model.eta.iter().enumerate() {
        let comma = if idx + 1 == params.model.eta.len() {
            ""
        } else {
            ","
        };
        writeln!(out, "    {eta}{comma}")?;
    }
    writeln!(out, "  ],")?;
    writeln!(out, "  \"birth_rate\": {},", params.tree_rate)?;
    writeln!(out, "  \"tree_shape\": {},", params.tree_shape)?;
    writeln!(out, "  \"tree_rate\": {},", params.tree_rate)?;
    writeln!(out, "  \"t_max\": {},", params.t_max)?;
    writeln!(out, "  \"root_length\": {},", root_len)?;
    writeln!(out, "  \"leaf_count\": {},", leaf_count)?;
    writeln!(out, "  \"max_tape_len\": {},", params.model.m)?;
    writeln!(out, "  \"num_tapes\": 1,")?;
    writeln!(out, "  \"symmetric\": false,")?;
    writeln!(out, "  \"rho\": {},", params.model.rho)?;
    writeln!(out, "  \"dt\": {},", params.model.dt)?;
    writeln!(out, "  \"tree_seed\": {},", params.tree_seed)?;
    writeln!(out, "  \"edit_seed\": {},", params.edit_seed)?;
    writeln!(out, "  \"sample_seed\": {}", params.sample_seed)?;
    writeln!(out, "}}")?;

    Ok(())
}

fn print_usage() {
    println!(
        "usage: cargo run -- simulate [--out simulations/30] [--seed 30] \\
         [--shape 2.0] [--rate 0.5] [--t-max 20.0] [--rho 0.1] \\
         [--tau 0.15] [--lambda 1.0] [--eta 0.1,0.1,0.1] [--m 8] [--dt 0.05]"
    );
}
