# lazy_smartcore

A "lazy predict" style CLI for Rust, built on [smartcore](https://smartcorelib.org/).
Point it at a CSV file, and it trains a battery of classifiers or regressors,
scores them on a held-out split, and prints a sorted leaderboard — similar in
spirit to Python's `lazypredict` package.

## Build

```bash
cargo build --release
```

## Usage

```bash
./target/debug/lazy_smartcore -h
```

### Options
```

| |       __ _   ____  _   _  |  _ \   _   _   ___  | |_ 
| |      / _` | |_  / | | | | | |_) | | | | | / __| | __|
| |___  | (_| |  / /  | |_| | |  _ <  | |_| | \__ \ | |_ 
|_____|  \__,_| /___|  \__, | |_| \_\  \__,_| |___/  \__|
                      |___/                             

All classifier and Regressor for AI/ML
     ************************************************
     Gaurav Sablok,
     Email: gsablok@proton.me
    ************************************************

Usage: lazy_smartcore [OPTIONS] --input <FILE>

Options:
-i, --input <FILE>           Path to the input CSV file
-t, --target <TARGET>        Name of the target column. Defaults to the last column if not found [default: target]
-k, --task <TASK>            Task type: classification or regression [default: classification] [possible values: classification, regression]
    --test-size <TEST_SIZE>  Fraction of data held out for testing (0.0-1.0) [default: 0.3]
    --no-shuffle             Disable shuffling before the train/test split
    --no-header              CSV file has no header row
-h, --help                   Print help
-V, --version                Print version
```

### Examples

Classification (target column named `target`):

```bash
lazy_smartcore --input iris.csv --target target --task classification
```

Regression, 20% test split, no shuffling:

```bash
lazy_smartcore --input housing.csv --target price --task regression \
  --test-size 0.2 --no-shuffle
```

### Sample output

```
Loaded 20 rows, 4 features from iris.csv
Features: sepal_length, sepal_width, petal_length, petal_width

=== Classification Leaderboard ===
Model                  Accuracy     Time (s)
--------------------------------------------
KNN                      1.0000       0.0000
LogisticRegression       1.0000       0.0074
DecisionTree             0.6667       0.0000
GaussianNB                0.6667       0.0000
RandomForest              0.5000       0.0010
```

## Models included

**Classification:** KNN, DecisionTree, RandomForest, LogisticRegression,
GaussianNB, SVC (linear kernel — binary classes only in this smartcore
version).

**Regression:** LinearRegression, Ridge, Lasso, ElasticNet, DecisionTree,
RandomForest, KNN.

## Notes / limitations

- All numeric columns are expected to already be numeric in the CSV (no
  built-in categorical encoding).
- No feature scaling is applied automatically — KNN/SVC in particular will
  benefit from standardizing inputs yourself beforehand.
- The bundled `smartcore` SVC implementation only supports binary
  classification; it will report a fit failure (non-fatal, skipped in the
  leaderboard) on datasets with 3+ classes.
