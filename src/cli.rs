use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Copy, Clone, PartialEq, Eq, Debug, ValueEnum)]
pub enum Task {
    Classification,
    Regression,
}

/// Lazy Predict for Rust, powered by smartcore.
/// Trains a battery of models on a CSV dataset and prints a leaderboard.
#[derive(Parser, Debug)]
#[command(
    name = "lazyRust",
    version = "1.0",
    about = "All classifier and Regressor for AI/ML
       ************************************************
       Gaurav Sablok,
       Email: gsablok@proton.me
      ************************************************"
)]
pub struct Cli {
    /// Path to the input CSV file
    #[arg(short, long, value_name = "FILE")]
    pub input: PathBuf,

    /// Name of the target column. Defaults to the last column if not found.
    #[arg(short, long, default_value = "target")]
    pub target: String,

    /// Task type: classification or regression
    #[arg(short = 'k', long, value_enum, default_value_t = Task::Classification)]
    pub task: Task,

    /// Fraction of data held out for testing (0.0-1.0)
    #[arg(long, default_value_t = 0.3)]
    pub test_size: f32,

    /// Disable shuffling before the train/test split
    #[arg(long, default_value_t = false)]
    pub no_shuffle: bool,

    /// CSV file has no header row
    #[arg(long, default_value_t = false)]
    pub no_header: bool,
}
