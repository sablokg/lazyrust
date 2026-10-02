mod classifier;
mod cli;
mod data;
mod regressor;
use anyhow::Result;
use clap::Parser;
use classifier::LazyClassifier;
use cli::{Cli, Task};
use data::load_csv;
use figlet_rs::FIGfont;
use regressor::LazyRegressor;

/*
Gaurav Sablok
gsablok@proton.me
*/

fn main() -> Result<()> {
    let fontgenerate = FIGfont::standard().unwrap();
    let repgenerate = fontgenerate.convert("LazyRust");
    println!("{}", repgenerate.unwrap());

    let args = Cli::parse();

    let loaded = load_csv(&args.input, &args.target, !args.no_header)?;

    println!(
        "Loaded {} rows, {} features from {}",
        loaded.y_raw.len(),
        loaded.feature_names.len(),
        args.input.display()
    );
    println!("Features: {}\n", loaded.feature_names.join(", "));

    match args.task {
        Task::Classification => {
            let y: Vec<u32> = loaded.y_raw.iter().map(|v| *v as u32).collect();
            let engine = LazyClassifier {
                test_size: args.test_size,
                shuffle: !args.no_shuffle,
            };
            let results = engine.fit_predict(&loaded.x, &y);
            println!("=== Classification Leaderboard ===");
            LazyClassifier::print_leaderboard(&results);
        }
        Task::Regression => {
            let engine = LazyRegressor {
                test_size: args.test_size,
                shuffle: !args.no_shuffle,
            };
            let results = engine.fit_predict(&loaded.x, &loaded.y_raw);
            println!("=== Regression Leaderboard ===");
            LazyRegressor::print_leaderboard(&results);
        }
    }

    Ok(())
}
