use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::elastic_net::ElasticNet;
use smartcore::linear::lasso::Lasso;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::linear::ridge_regression::RidgeRegression;
use smartcore::metrics::{mean_absolute_error, mean_squared_error};
use smartcore::model_selection::train_test_split;
use smartcore::neighbors::knn_regressor::KNNRegressor;
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressor;
use std::time::Instant;

/*
Gaurav Sablok
gsablok@proton.me
*/

pub struct ModelResult {
    pub name: &'static str,
    pub rmse: f64,
    pub mae: f64,
    pub time_secs: f64,
}

pub struct LazyRegressor {
    pub test_size: f32,
    pub shuffle: bool,
}

impl Default for LazyRegressor {
    fn default() -> Self {
        LazyRegressor {
            test_size: 0.3,
            shuffle: true,
        }
    }
}

impl LazyRegressor {
    pub fn fit_predict(&self, x: &DenseMatrix<f64>, y: &Vec<f64>) -> Vec<ModelResult> {
        let (x_train, x_test, y_train, y_test) =
            train_test_split(x, y, self.test_size, self.shuffle, None);

        let mut results = Vec::new();

        macro_rules! run {
            ($name:expr, $fit:expr) => {{
                let start = Instant::now();
                match $fit {
                    Ok(model) => match model.predict(&x_test) {
                        Ok(preds) => {
                            let mse = mean_squared_error(&y_test, &preds);
                            let mae = mean_absolute_error(&y_test, &preds);
                            results.push(ModelResult {
                                name: $name,
                                rmse: mse.sqrt(),
                                mae,
                                time_secs: start.elapsed().as_secs_f64(),
                            });
                        }
                        Err(e) => eprintln!("{} predict failed: {:?}", $name, e),
                    },
                    Err(e) => eprintln!("{} fit failed: {:?}", $name, e),
                }
            }};
        }

        run!(
            "LinearRegression",
            LinearRegression::fit(&x_train, &y_train, Default::default())
        );
        run!(
            "Ridge",
            RidgeRegression::fit(&x_train, &y_train, Default::default())
        );
        run!("Lasso", Lasso::fit(&x_train, &y_train, Default::default()));
        run!(
            "ElasticNet",
            ElasticNet::fit(&x_train, &y_train, Default::default())
        );
        run!(
            "DecisionTree",
            DecisionTreeRegressor::fit(&x_train, &y_train, Default::default())
        );
        run!(
            "RandomForest",
            RandomForestRegressor::fit(&x_train, &y_train, Default::default())
        );
        run!(
            "KNN",
            KNNRegressor::fit(&x_train, &y_train, Default::default())
        );

        results.sort_by(|a, b| a.rmse.partial_cmp(&b.rmse).unwrap());
        results
    }

    pub fn print_leaderboard(results: &[ModelResult]) {
        println!(
            "{:<20} {:>10} {:>10} {:>12}",
            "Model", "RMSE", "MAE", "Time (s)"
        );
        println!("{}", "-".repeat(56));
        for r in results {
            println!(
                "{:<20} {:>10.4} {:>10.4} {:>12.4}",
                r.name, r.rmse, r.mae, r.time_secs
            );
        }
    }
}
