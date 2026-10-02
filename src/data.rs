use anyhow::{anyhow, Context, Result};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::path::Path;

/*
Gaurav Sablok
gsablok@proton.me
*/

/// Result of loading a CSV file split into features (x) and target (y).
pub struct LoadedData {
    pub x: DenseMatrix<f64>,
    pub y_raw: Vec<f64>,
    pub feature_names: Vec<String>,
}

/// Reads a CSV file, splits out the target column (by name, or the last
/// column if the name isn't found / there's no header), and returns
/// the remaining columns as a smartcore DenseMatrix plus the target as f64.
pub fn load_csv(path: &Path, target_col: &str, has_header: bool) -> Result<LoadedData> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(has_header)
        .from_path(path)
        .with_context(|| format!("failed to open CSV at {}", path.display()))?;

    let headers: Vec<String> = if has_header {
        reader
            .headers()
            .context("failed to read CSV header row")?
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        Vec::new()
    };

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for result in reader.records() {
        let record = result.context("failed to parse CSV row")?;
        let row: Result<Vec<f64>> = record
            .iter()
            .map(|field| {
                field
                    .trim()
                    .parse::<f64>()
                    .with_context(|| format!("could not parse '{field}' as a number"))
            })
            .collect();
        rows.push(row?);
    }

    if rows.is_empty() {
        return Err(anyhow!("CSV file at {} has no data rows", path.display()));
    }

    let num_cols = rows[0].len();
    for (i, row) in rows.iter().enumerate() {
        if row.len() != num_cols {
            return Err(anyhow!(
                "row {} has {} columns, expected {}",
                i + 1,
                row.len(),
                num_cols
            ));
        }
    }

    // Figure out which column index is the target.
    let target_idx = if has_header {
        headers
            .iter()
            .position(|h| h == target_col)
            .unwrap_or(num_cols - 1)
    } else {
        num_cols - 1
    };

    let feature_names: Vec<String> = (0..num_cols)
        .filter(|&i| i != target_idx)
        .map(|i| {
            if has_header {
                headers[i].clone()
            } else {
                format!("col_{i}")
            }
        })
        .collect();

    let mut x_rows: Vec<Vec<f64>> = Vec::with_capacity(rows.len());
    let mut y_raw: Vec<f64> = Vec::with_capacity(rows.len());

    for row in rows {
        y_raw.push(row[target_idx]);
        let features: Vec<f64> = row
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != target_idx)
            .map(|(_, v)| *v)
            .collect();
        x_rows.push(features);
    }

    let x_slices: Vec<&[f64]> = x_rows.iter().map(|r| r.as_slice()).collect();
    let x = DenseMatrix::from_2d_array(&x_slices)
        .map_err(|e| anyhow!("failed to build feature matrix: {e}"))?;

    Ok(LoadedData {
        x,
        y_raw,
        feature_names,
    })
}
