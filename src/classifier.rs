use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::svm::svc::{SVCParameters, SVC};
use smartcore::svm::Kernels;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use std::time::Instant;

/*
Gaurav Sablok
gsablok@proton.me
*/

pub struct ModelResult {
    pub name: &'static str,
    pub accuracy: f64,
    pub time_secs: f64,
}

pub struct LazyClassifier {
    pub test_size: f32,
    pub shuffle: bool,
}

impl Default for LazyClassifier {
    fn default() -> Self {
        LazyClassifier {
            test_size: 0.3,
            shuffle: true,
        }
    }
}

impl LazyClassifier {
    pub fn fit_predict(&self, x: &DenseMatrix<f64>, y: &Vec<u32>) -> Vec<ModelResult> {
        let (x_train, x_test, y_train, y_test) =
            train_test_split(x, y, self.test_size, self.shuffle, None);

        let mut results = Vec::new();

        macro_rules! run {
            ($name:expr, $fit:expr) => {{
                let start = Instant::now();
                match $fit {
                    Ok(model) => match model.predict(&x_test) {
                        Ok(preds) => {
                            let acc = accuracy(&y_test, &preds);
                            results.push(ModelResult {
                                name: $name,
                                accuracy: acc,
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
            "KNN",
            KNNClassifier::fit(&x_train, &y_train, Default::default())
        );
        run!(
            "DecisionTree",
            DecisionTreeClassifier::fit(&x_train, &y_train, Default::default())
        );
        run!(
            "RandomForest",
            RandomForestClassifier::fit(&x_train, &y_train, Default::default())
        );
        run!(
            "LogisticRegression",
            LogisticRegression::fit(&x_train, &y_train, Default::default())
        );
        run!(
            "GaussianNB",
            GaussianNB::fit(&x_train, &y_train, Default::default())
        );
        {
            // SVC::predict returns Vec<TX> (the feature type, f64) rather than
            // the label type, so its output needs to be cast back to u32
            // before it can be scored against y_test.
            let start = Instant::now();
            let svc_params = SVCParameters::default().with_kernel(Kernels::linear());
            match SVC::fit(&x_train, &y_train, &svc_params) {
                Ok(model) => match model.predict(&x_test) {
                    Ok(preds_f64) => {
                        let preds: Vec<u32> = preds_f64.iter().map(|v| v.round() as u32).collect();
                        let acc = accuracy(&y_test, &preds);
                        results.push(ModelResult {
                            name: "SVC",
                            accuracy: acc,
                            time_secs: start.elapsed().as_secs_f64(),
                        });
                    }
                    Err(e) => eprintln!("SVC predict failed: {:?}", e),
                },
                Err(e) => eprintln!("SVC fit failed: {:?}", e),
            }
        }

        results.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap());
        results
    }

    pub fn print_leaderboard(results: &[ModelResult]) {
        println!("{:<20} {:>10} {:>12}", "Model", "Accuracy", "Time (s)");
        println!("{}", "-".repeat(44));
        for r in results {
            println!("{:<20} {:>10.4} {:>12.4}", r.name, r.accuracy, r.time_secs);
        }
    }
}
