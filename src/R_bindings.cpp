// [[Rcpp::interfaces(r, cpp)]]
// [[Rcpp::depends(RcppEigen)]]
#include <chrono>
#include <Rcpp.h>
#include <RcppEigen.h>
#include "lib.hpp"

using namespace Rcpp;

MatrixXd mat_from_df(const DataFrame& df, const StringVector& sv)
{
    MatrixXd m(sv.size(), df.nrows());
    for (int i = 0; i < sv.size(); i++) {
        m.row(i) = as<VectorXd>(df[as<std::string>(sv(i))]); 
    }
    return m;
}

DataFrame export_score(const Model& model)
{   const ScoreParams& params = model.get_params();
    return DataFrame::create(Named("cst") = params.y0, 
                             Named("w") = params.w,
                             Named("w1") = model.w1,
                             Named("w2") = model.w2,
                             Named("idx") = model.get_idxs(),
                             Named("split_val") = model.get_split_val()
                            );
}

std::unordered_map<int, VectorXd>
build_quantile_map(const MatrixXd& m,
                   int n_quantiles,
                   Rcpp::Nullable<Rcpp::List> user_quantiles)
{
    int n_features = m.rows();
    std::unordered_map<int, VectorXd> qts;

    if (user_quantiles.isNotNull()) {
        Rcpp::List qlist(user_quantiles);

        if (qlist.size() != n_features) {
            Rcpp::stop("Length of quantiles list must match number of features (columns in x)");
        }

        for (int i = 0; i < n_features; ++i) {
            if (qlist[i] != R_NilValue) {
                Rcpp::NumericVector qv = qlist[i];
                qts[i] = Rcpp::as<VectorXd>(qv);
            } else {
                VectorXd col = m.row(i);
                qts[i] = quantiles(col, n_quantiles);
            }
        }
    } else {
        qts = make_quantiles(m, n_quantiles);
    }

    return qts;
}

// [[Rcpp::export]]
DataFrame fit_proba(NumericVector x, NumericVector y, int n_iter, double lr, int n_quantiles, int batch_size)
{
    MatrixXd m = as<MatrixXd>(x).transpose(); // TODO: can we avoid this copy?
    VectorXd yv = as<VectorXd>(y); // TODO: we can also avoid this copy
    double lo = logodds(yv);
    const std::unordered_map<int, VectorXd> qts = make_quantiles(m, n_quantiles);
    Model model(m.rows(), m.cols(), n_iter, lr, n_quantiles, batch_size);
    model.params.y0 = lo;
    model.fit_proba(m, yv, qts);
    return export_score(model);
}

// [[Rcpp::export]]
DataFrame fit(NumericVector x, NumericVector y, int n_iter, double lr, int n_quantiles, int batch_size)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    MatrixXd m = as<MatrixXd>(x).transpose(); // TODO: can we avoid this copy?
    VectorXd yv = as<VectorXd>(y); // TODO: we can also avoid this copy

    auto t2 = std::chrono::high_resolution_clock::now();
    const std::unordered_map<int, VectorXd> qts = make_quantiles(m, n_quantiles);
    Model model(m.rows(), m.cols(), n_iter, lr, n_quantiles, batch_size);
    model.fit(m, yv, qts);

    auto t3 = std::chrono::high_resolution_clock::now();
    auto dur1 = std::chrono::duration_cast<std::chrono::milliseconds> (t3 - t1);
    auto dur2 = std::chrono::duration_cast<std::chrono::milliseconds> (t3 - t2);
    //Rcout << "t1 : " << dur1.count() << "  t2 : " << dur2.count() << "\n";
    return export_score(model);
}

// [[Rcpp::export]]
DataFrame fit_survival(NumericMatrix x,
                       NumericVector time,
                       NumericVector event,
                       int n_iter,
                       double lr,
                       int n_quantiles,
                       int batch_size,
                       Rcpp::Nullable<Rcpp::List> quantiles) {
    using namespace std::chrono;

    const auto start_total = high_resolution_clock::now();

    // Convert input to Eigen
    Eigen::MatrixXd m = as<MatrixXd>(x).transpose();  // shape: features x samples
    Eigen::VectorXd T = as<VectorXd>(time);
    Eigen::VectorXd E = as<VectorXd>(event);

    // Precompute quantiles (split thresholds)
    const auto qts = build_quantile_map(m, n_quantiles, quantiles);

    // Create and fit model
    Model model(m.rows(), m.cols(), n_iter, lr, n_quantiles, batch_size);

    const auto start_fit = high_resolution_clock::now();
    model.fit_survival(m, T, E, qts);
    const auto end_fit = high_resolution_clock::now();

    // Timing output
    const auto total_duration = duration_cast<milliseconds>(end_fit - start_total).count();
    const auto fit_duration = duration_cast<milliseconds>(end_fit - start_fit).count();
    //Rcpp::Rcout << "Total time: " << total_duration << " ms, "
    //            << "Fit time: " << fit_duration << " ms\n";

    // Export model state to R
    return export_score(model);
}
