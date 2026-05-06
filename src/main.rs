fn main() {
    if let Err(err) = sequential_editing_likelihood::simulate::run_from_args() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
