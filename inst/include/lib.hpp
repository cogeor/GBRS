#pragma once
#include <vector>
#include <numeric>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include <omp.h>

using Eigen::Map;                      
using Eigen::MatrixXd;                 
using Eigen::VectorXd;
using Eigen::VectorXi;

double sum_p_complement(VectorXd& p);
double logodds(const VectorXd& grps);
void convert_logodds_to_p_inplace(VectorXd& p);
std::unordered_map<int, VectorXd> make_quantiles(const MatrixXd& x, const int n_pts);
VectorXd quantiles(const VectorXd & v, const int n_quantiles);


struct ScoreParams {
    double y0;
    VectorXd w;
};

class Model
{
private:
    int max_n;
    double lr;
    int batch_size;
    int n_interp;
    int i;
    VectorXd idxs;
    VectorXd split_val;
    MatrixXd iter_out;
    VectorXd mask;
    VectorXd l_arr;
    MatrixXd gmat;
    //std::unordered_map<int, Eigen::VectorXd> mean_bins;
    void add_elem(const int idx, const double sv, const double w1, const double w2);
    void iter(const MatrixXd& x, const VectorXd& y, const std::unordered_map<int, VectorXd>& qtsw);
    void iter_proba(const MatrixXd& x, const VectorXd& res, const VectorXd& y, const std::unordered_map<int, VectorXd>& qtsw);
    void iter_survival(const MatrixXd& x,
                       const VectorXd& T,
                       const VectorXd& E,
                       const std::unordered_map<int, VectorXd>& qtsw,
                       const VectorXd& f);
    std::array<double, 4> get_best_split(const VectorXd& u, const VectorXd& y, const VectorXd& interp_pts);
    std::array<double, 4> get_best_split_proba(const VectorXd& u, const VectorXd& preds, const VectorXd& y, const VectorXd& interp_pts);
    std::array<double, 4> get_best_split_survival(
        const VectorXd& u,               
        const VectorXd& gradients,       
        const VectorXd& f,               
        const VectorXd& T,               
        const VectorXd& E,               
        const VectorXd& interp_pts);
    
    // Helper functions for subsampling
    std::vector<int> get_batch_indices(int n_samples) const {
        if (batch_size <= 0 || batch_size >= n_samples) {
            return {}; // Empty implies full dataset
        }
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
        indices.resize(batch_size);
        return indices;
    }

    MatrixXd slice_matrix(const MatrixXd& m, const std::vector<int>& idxs) const {
        if (idxs.empty()) return m; // Should not happen with current logic but safe check
        MatrixXd sub(idxs.size(), m.cols());
        for (size_t i = 0; i < idxs.size(); ++i) {
            sub.row(i) = m.row(idxs[i]);
        }
        return sub;
    }

    VectorXd slice_vector(const VectorXd& v, const std::vector<int>& idxs) const {
        if (idxs.empty()) return v;
        VectorXd sub(idxs.size());
        for (size_t i = 0; i < idxs.size(); ++i) {
            sub[i] = v[idxs[i]];
        }
        return sub;
    }

public:
    //std::map<int, VectorXd> qts;
    ScoreParams params;
    VectorXd w1;
    VectorXd w2;
    Model(int nrow, int ncol, int max_n, double lr, int n_pts, int batch_size)
    : max_n(max_n)
    , iter_out(MatrixXd(nrow, 4))
    , mask(VectorXd(nrow))
    , l_arr(VectorXd(n_pts))
    , gmat(MatrixXd(n_pts, 2))
    , lr(lr)
    , batch_size(batch_size)
    , idxs(VectorXd::Zero(max_n))
    , split_val(VectorXd::Zero(max_n))
    , w1(VectorXd::Zero(max_n))
    , w2(VectorXd::Zero(max_n))
    , i(0)
    , params({0.0, VectorXd::Zero(max_n)})
    {}
    VectorXd predict(const MatrixXd& x) const;
    VectorXd predict_lin_interp(const MatrixXd& x) const;
    VectorXd predict_proba(const MatrixXd& x) const;
    VectorXd predict_debug(const MatrixXd& x, const double y0) const;
    void fit(const MatrixXd& m, const VectorXd& y, const std::unordered_map<int, VectorXd>& qtsw);
    void fit_proba(const MatrixXd& m, const VectorXd& y, const std::unordered_map<int, VectorXd>& qtsw);
    void fit_survival(const MatrixXd& X,
                             const VectorXd& T,
                             const VectorXd& E,
                             const std::unordered_map<int, VectorXd>& qts);
    MatrixXd export_model() const;
    const ScoreParams& get_params() const;
    const VectorXd& get_idxs() const;
    const VectorXd& get_split_val() const;
    void prune();
    void set_params(const VectorXd& idxs, const VectorXd& split_val, const VectorXd& w, const double y0);
};


VectorXd convert_logodds_to_p(const VectorXd& logodds) 
{
    int n = logodds.size();
    VectorXd out(n);
    for (int i = 0; i < n; i++) {
        out[i] = std::exp(logodds[i]) / (1 + std::exp(logodds[i]));
    }
    return out;
}

void convert_logodds_to_p_inplace(VectorXd& p) 
{
    int n = p.size();
    for (int i = 0; i < n; i++) {
        p[i] = std::exp(p[i]) / (1 + std::exp(p[i]));
    }
}

double logodds(const VectorXd& grps)
{
    double s1 = 0;
    double n = grps.size();
    for (int i = 0; i < grps.size(); i++) {
        s1 += grps[i];
    }
    return std::log(s1 / (n - s1));
}

VectorXd quantiles(const VectorXd & v, const int n_quantiles)
{
    VectorXd out(n_quantiles);
    std::vector<double> sorted(v.data(), v.data() + v.size()); // need to make a copy to sort
    int n_samples = v.size();
    std::sort(sorted.begin(), sorted.end());

    for (int i = 0; i < n_quantiles; i++) {
        int idx = static_cast<int>(static_cast<double>(i + 1) / (n_quantiles + 1) * (n_samples - 1));
        out[i] = sorted[idx];
    } 
    return out;
}

VectorXd less_than(const VectorXd& v, const double val)
{
    VectorXd out(v.size());
    for (int i = 0; i < v.size(); i++) {
        out[i] = v[i] < val ? 1.0 : 0.0;
    }
    return out;
}

VectorXd greater_than(const VectorXd& v, const double val)
{
    VectorXd out(v.size());
    for (int i = 0; i < v.size(); i++) {
        out[i] = v[i] > val ? 1.0 : 0.0;
    }
    return out;
}

std::array<double, 2> gamma(const VectorXd& grps, const VectorXd& y) 
{
    std::array<double, 2> g{0.0, 0.0};
    int n = grps.size();
    int s1 = 0, s2 = 0; 
    for (size_t i = 0; i < n; i++) {
        if (grps[i] == 0.0) {
            s1 += 1;
            g[0] += y[i];
        } else {
            s2 += 1;
            g[1] += y[i];
        }
    }
    if (s1 == 0) {
        s1 = 1;
    } 
    if (s2 == 0) {
        s2 = 1;
    } 
    g[0] /= s1;
    g[1] /= s2;

    return g;
}

VectorXd gamma_logodds(const VectorXd& grps, const VectorXd& y_pred, const VectorXd& p_true) 
{
    VectorXd g = VectorXd::Zero(2);
    VectorXd p_y = convert_logodds_to_p(y_pred);
    VectorXd res = p_true - p_y;
    int n = grps.size();
    double s1 = 0, s2 = 0; 
    for (size_t i = 0; i < n; i++) {
        if (grps[i] == 0.0) {
            g[0] += res[i];
            s1 += p_y[i] * (1 - p_y[i]);
        } else {
            g[1] += res[i];
            s2 += p_y[i] * (1 - p_y[i]);
        }
    }
    if (s1 == 0) {
        s1 = 1;
    } 
    if (s2 == 0) {
        s2 = 1;
    } 
    g[0] /= s1;
    g[1] /= s2;

    return g;
}

std::array<double, 2> gamma_survival(const VectorXd& grps, const VectorXd& gradients) {
    std::array<double, 2> g{0.0, 0.0};
    int s1 = 0, s2 = 0;
    for (int i = 0; i < grps.size(); i++) {
        if (grps[i] == 0.0) {
            s1++;
            g[0] += gradients[i];
        } else {
            s2++;
            g[1] += gradients[i];
        }
    }
    if (s1 == 0) s1 = 1;
    if (s2 == 0) s2 = 1;
    g[0] /= s1;
    g[1] /= s2;
    return g;
}

double l2_norm_mask(const VectorXd& y_p, const VectorXd& y, const VectorXd& ss_mask)
{
    double sum = 0;
    //Rcout << "a " << y_p.size() << "b "<<y.size() << "\n";  
    for (size_t i = 0; i < y.size(); i++) {
        sum += (y_p[i] - y[i]) * (y_p[i] - y[i]) * ss_mask[i];
    }
    return sum;
}

double l2_norm(const VectorXd& y_p, const VectorXd& y)
{
    double sum = 0;
    //Rcout << "a " << y_p.size() << "b "<<y.size() << "\n";  
    for (size_t i = 0; i < y.size(); i++) {
        sum += (y_p[i] - y[i]) * (y_p[i] - y[i]);
    }
    return sum;
}

VectorXd weight_groups(const VectorXd& mask, const double w1, const double w2)
{
    VectorXd vec(mask.size());
    const int n = mask.size();
    for (int i = 0; i < n; i++) {
        vec[i] = (mask[i] == 0.0) ? w1 : w2;
    }
    return vec;
}

void weight_groups_inplace(VectorXd& vec, const VectorXd& mask, const double w1, const double w2)
{
    const int n = mask.size();
    for (int i = 0; i < n; i++) {
        vec[i] += (mask[i] == 0.0) ? w1 : w2;
    }
}

void set_weighted_predictions(VectorXd& vec, const VectorXd& mask, const double w1, const double w2)
{
    const int n = mask.size();
    for (int i = 0; i < n; i++) {
        vec[i] = (mask[i] == 0.0) ? w1 : w2;
    }
}

double cross_entropy_norm(const VectorXd& y_pred, const VectorXd& y_true)
{
    VectorXd clipped_y_pred = y_pred.cwiseMax(1e-15).cwiseMin(1 - 1e-15);
//    double cross_entropy = -(y_true.array() * clipped_y_pred.array().log() + 
//                             (1 - y_true.array()) * (1 - clipped_y_pred.array()).log()).sum();
//
    double cross_entropy = -(y_true.array() * clipped_y_pred.array().log()).sum();
    return cross_entropy;
}

double ranking_loss(const VectorXd& f,
                    const VectorXd& T,
                    const VectorXd& E,
                    const VectorXd& mask) {
    const auto n = f.size();
    double loss = 0.0;
    int pair_count = 0;

    for (Eigen::Index i = 0; i < n; ++i) {
        if (E[i] != 1 || mask[i] != 1) continue; 

        for (Eigen::Index j = 0; j < n; ++j) {
            if (mask[j] != 1) continue;

            if (T[i] < T[j]) {
                double diff = f[j] - f[i];
                loss += std::log(1 + std::exp(diff));
                pair_count++;
            }
        }
    }

    if (pair_count == 0) return 0.0;
    return loss / pair_count;
}


VectorXd compute_ranking_gradients_par(const VectorXd& f,
                                   const VectorXd& T,
                                   const VectorXd& E,
                                   const VectorXd& mask) {
    const auto n = f.size();
    const int n_threads = omp_get_max_threads();

    // Thread-local gradient buffers
    std::vector<VectorXd> grad_locals(n_threads, VectorXd::Zero(n));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        VectorXd& grad = grad_locals[tid];

        #pragma omp for schedule(static)
        for (Eigen::Index i = 0; i < n; ++i) {
            if (E[i] != 1 || mask[i] != 1) continue;

            for (Eigen::Index j = 0; j < n; ++j) {
                if (mask[j] != 1) continue;

                if (T[i] < T[j]) {
                    double diff = f[j] - f[i];
                    double exp_diff = std::exp(diff);
                    double sigmoid = exp_diff / (1.0 + exp_diff);

                    grad[i] -= sigmoid;
                    grad[j] += sigmoid;
                }
            }
        }
    }

    // Combine gradients
    VectorXd grad_final = VectorXd::Zero(n);
    for (const auto& g : grad_locals) {
        grad_final += g;
    }

    return -grad_final;
}

VectorXd compute_ranking_gradients(const VectorXd& f,
                                  const VectorXd& T,
                                  const VectorXd& E,
                                  const VectorXd& mask) {
    const auto n = f.size();
    VectorXd grad = VectorXd::Zero(n);

    for (Eigen::Index i = 0; i < n; ++i) {
        if (E[i] != 1 || mask[i] != 1) continue; 

        for (Eigen::Index j = 0; j < n; ++j) {
            if (mask[j] != 1) continue;

            if (T[i] < T[j]) {
                double diff = f[j] - f[i];
                double exp_diff = std::exp(diff);
                double sigmoid = exp_diff / (1.0 + exp_diff);

                grad[i] -= sigmoid;
                grad[j] += sigmoid;
            }
        }
    }

    return -grad;
}

double cox_partial_ll_loss(const VectorXd& f,
                           const VectorXd& T,
                           const VectorXd& E,
                           const VectorXd& mask) {
    const auto n = f.size();
    const VectorXd exp_f = f.array().exp();
    double loss = 0.0;

    for (Eigen::Index i = 0; i < n; ++i) {
        if (E[i] == 1 && mask[i] == 1) {
            double denom = 0.0;

            for (Eigen::Index j = 0; j < n; ++j) {
                if (T[j] >= T[i] && mask[j] == 1) {
                    denom += exp_f[j];
                }
            }

            loss -= (f[i] - std::log(denom));
        }
    }
    return loss;
}

VectorXd compute_cox_gradients(const VectorXd& f, const VectorXd& T, const VectorXd& E) {
    int n = f.size();
    VectorXd grad = VectorXd::Zero(n);
    VectorXd exp_f = f.array().exp();

    for (int i = 0; i < n; ++i) {
        if (E[i] == 1) {
            double denom = 0.0;
            for (int j = 0; j < n; ++j) {
                if (T[j] >= T[i]) {
                    denom += exp_f[j];
                }
            }
            grad[i] = 1.0 - (exp_f[i] / denom);
        } else {
            grad[i] = 0.0;  // No contribution from censored samples directly
        }
    }

    return grad;  // Negative gradient (pseudo-residual)
}

double cross_entropy_norm_mask(const VectorXd& y_pred, const VectorXd& y_true, const VectorXd& ss_mask)
{
    VectorXd clipped_y_pred = y_pred.cwiseMax(1e-15).cwiseMin(1 - 1e-15);
//    double cross_entropy = -(y_true.array() * clipped_y_pred.array().log() + 
//                             (1 - y_true.array()) * (1 - clipped_y_pred.array()).log()).sum();
//
    double cross_entropy = -(ss_mask.array() * y_true.array() * clipped_y_pred.array().log()).sum();
    return cross_entropy;
}

std::array<double, 4> Model::get_best_split(const VectorXd& u, const VectorXd& y, const VectorXd& interp_pts)
{
    size_t n_pts = interp_pts.size();
    std::array<double, 2> g_tmp;
    std::array<double, 4> out;
    VectorXd new_preds(u.size());
    // Thread-local variables to avoid race conditions
    VectorXd l_arr(n_pts);
    MatrixXd gmat(n_pts, 2);
    
    for (int i = 0; i < n_pts; i++) {
        VectorXd mask = greater_than(u, interp_pts[i]);
        g_tmp = gamma(mask, y);
        gmat(i, 0) = g_tmp[0]; 
        gmat(i, 1) = g_tmp[1]; 
        set_weighted_predictions(new_preds, mask, lr * g_tmp[0], lr * g_tmp[1]);
        l_arr[i] = l2_norm(new_preds, y);
    }
    int idx = 0;
    out[0] = l_arr.minCoeff(&idx);
    out[1] = interp_pts[idx];
    out[2] = gmat(idx, 0);
    out[3] = gmat(idx, 1);
    return out;
}

std::array<double, 4> Model::get_best_split_proba(const VectorXd& u, const VectorXd& preds, const VectorXd& y, const VectorXd& interp_pts)
{
    int n_pts = interp_pts.size();
    int idx = 0;
    VectorXd new_preds(u.size());
    std::array<double, 4> out;
    // Thread-local variables to avoid race conditions
    VectorXd l_arr(n_pts);
    MatrixXd gmat(n_pts, 2);
    
    for (int i = 0; i < n_pts; i++) {
        VectorXd mask = greater_than(u, interp_pts[i]);
        gmat.row(i) = gamma_logodds(mask, preds, y);
        new_preds = preds;
        weight_groups_inplace(new_preds, mask, lr * gmat(i, 0), lr * gmat(i, 1));
        convert_logodds_to_p_inplace(new_preds);
        l_arr[i] = cross_entropy_norm(new_preds, y);
        //printf("l %f m %f \n", l_arr[i], mask.sum());
    }
    out[0] = l_arr.minCoeff(&idx);
    out[1] = interp_pts[idx];
    out[2] = gmat(idx, 0);
    out[3] = gmat(idx, 1);
    return out;
}

std::array<double, 4> Model::get_best_split_survival(
    const VectorXd& u,               
    const VectorXd& gradients,       
    const VectorXd& f,               
    const VectorXd& T,               
    const VectorXd& E,               
    const VectorXd& interp_pts       
) {
    const size_t n_pts = interp_pts.size();
    std::array<double, 2> g_tmp;
    std::array<double, 4> out;
    VectorXd l_arr(n_pts);
    MatrixXd gmat(n_pts, 2);
    VectorXd new_preds(u.size());

    for (int i = 0; i < n_pts; i++) {
        auto mask = greater_than(u, interp_pts[i]); // binary group mask
        g_tmp = gamma_survival(mask, gradients);

        gmat(i, 0) = g_tmp[0]; // left prediction
        gmat(i, 1) = g_tmp[1]; // right prediction

        set_weighted_predictions(new_preds, mask, lr * g_tmp[0], lr * g_tmp[1]);
        new_preds += f;

        l_arr[i] = ranking_loss(new_preds, T, E, VectorXd::Ones(E.size()));
    }

    int idx = 0;
    out[0] = l_arr.minCoeff(&idx);         // best loss
    out[1] = interp_pts[idx];              // best split threshold
    out[2] = gmat(idx, 0);                 // left group prediction
    out[3] = gmat(idx, 1);                 // right group prediction
    return out;
}

const VectorXd& Model::get_idxs() const
{
    return this->idxs;
}

const VectorXd& Model::get_split_val() const
{
    return this->split_val;
}

VectorXd Model::predict(const MatrixXd& x) const 
{
    VectorXd p(x.cols());
    p.setConstant(this->params.y0);
    VectorXd mask(x.cols());
    for (int s_idx = 0; s_idx < this->i; s_idx++) {
        mask = greater_than(x.row(this->idxs[s_idx]), this->split_val[s_idx]);
        //p += weight_groups(mask, this->w1[s_idx], this->w2[s_idx]);
        weight_groups_inplace(p, mask, 0.0, this->params.w[s_idx]);
    }
    return p;
}

double weighted_mean(const VectorXd& x, const VectorXd& mask) {
    double count = mask.sum();
    if (count == 0.0) return std::numeric_limits<double>::quiet_NaN();
    return (x.array() * mask.array()).sum() / count;
}

// Computes binned means for a single feature column
VectorXd compute_feature_bin_means(const VectorXd& x_col, const VectorXd& splits) {
    const int num_samples = x_col.size();
    const int num_bins = splits.size() + 1;
    VectorXd means(num_bins);

    for (int i = 0; i < num_bins; i++) {
        if (splits.size() == 2) {
            VectorXd means(2);

            means[0] = 0;
            means[1] = 1;

            return means;
        }

        VectorXd mask_lower, mask_upper, bin_mask;
        if (i == 0) {
            mask_lower = VectorXd::Ones(num_samples);
            mask_upper = (x_col.array() <= splits[0]).cast<double>();
        } else if (i == splits.size()) {
            mask_lower = (x_col.array() > splits[splits.size() - 1]).cast<double>();
            mask_upper = VectorXd::Ones(num_samples);
        } else {
            mask_lower = (x_col.array() > splits[i - 1]).cast<double>();
            mask_upper = (x_col.array() <= splits[i]).cast<double>();
        }

        bin_mask = mask_lower.array() * mask_upper.array();
        means[i] = weighted_mean(x_col, bin_mask);
    }

    return means;
}

// Main function: computes binned means for all specified features
std::unordered_map<int, VectorXd> compute_binned_feature_means(
    const MatrixXd& X,
    const std::unordered_map<int, VectorXd>& split_map)
{
    std::unordered_map<int, VectorXd> result;

    for (const auto& [feat_idx, splits] : split_map) {
        const VectorXd& x_col = X.col(feat_idx);
        result[feat_idx] = compute_feature_bin_means(x_col, splits);
    }

    return result;
}

VectorXd Model::predict_lin_interp(const MatrixXd& x) const 
{
    VectorXd p(x.cols());
    p.setConstant(this->params.y0);
    VectorXd mask(x.cols());
    for (int s_idx = 0; s_idx < this->i; s_idx++) {
        mask = greater_than(x.row(this->idxs[s_idx]), this->split_val[s_idx]);
        //p += weight_groups(mask, this->w1[s_idx], this->w2[s_idx]);
        weight_groups_inplace(p, mask, 0.0, this->params.w[s_idx]);
    }
    return p;
}

VectorXd Model::predict_proba(const MatrixXd& x) const 
{
    VectorXd p(x.cols());
    p.setConstant(this->params.y0);
    VectorXd mask(x.cols());
    for (int s_idx = 0; s_idx < this->i; s_idx++) {
        mask = greater_than(x.row(this->idxs[s_idx]), this->split_val[s_idx]);
        //p += weight_groups(mask, this->w1[s_idx], this->w2[s_idx]);
        weight_groups_inplace(p, mask, 0.0, this->params.w[s_idx]);
    }
    convert_logodds_to_p_inplace(p);
    return p;
}


VectorXd Model::predict_debug(const MatrixXd& x, const double y0) const 
{
    VectorXd p(x.cols());
    p.setConstant(y0);
    VectorXd mask(x.cols());
    for (int s_idx = 0; s_idx < this->i; s_idx++) {
        mask = greater_than(x.row(this->idxs[s_idx]), this->split_val[s_idx]);
        weight_groups_inplace(p, mask, this->w1[s_idx], this->w2[s_idx]);
    }
    //convert_logodds_to_p_inplace(p);
    return p;
}

void Model::add_elem(const int idx, const double sv, const double w1, const double w2)
{
        this->idxs[this->i] = idx;
        this->split_val[this->i] = sv;
        this->params.y0 += w2;
        this->params.w[this->i] = w2 - w1;
        this->w1[this->i] = w1;
        this->w2[this->i] = w2;
        this->i++;
}

void Model::iter(const MatrixXd& x, const VectorXd& y, const std::unordered_map<int, VectorXd>& qtsw)
{
    std::vector<int> indices = get_batch_indices(y.size());
    MatrixXd x_sub;
    VectorXd y_sub;
    const MatrixXd* x_ptr = &x; 
    const VectorXd* y_ptr = &y;

    if (!indices.empty()) {
        x_sub.resize(x.rows(), indices.size());
        y_sub.resize(indices.size());
        for(size_t k = 0; k < indices.size(); ++k) {
            x_sub.col(k) = x.col(indices[k]);
            y_sub[k] = y[indices[k]];
        }
        x_ptr = &x_sub;
        y_ptr = &y_sub;
    }

    #pragma omp parallel for shared(x_ptr, y_ptr, qtsw) schedule(dynamic)
    for (int i = 0; i < x_ptr->rows(); i++) {
        std::array<double, 4> out_tmp = this->get_best_split(x_ptr->row(i), *y_ptr, qtsw.at(i));
        this->iter_out(i, 0) = out_tmp[0];
        this->iter_out(i, 1) = out_tmp[1];
        this->iter_out(i, 2) = out_tmp[2];
        this->iter_out(i, 3) = out_tmp[3];
    }
    int min_idx = 0;
    this->iter_out.col(0).minCoeff(&min_idx);
    this->add_elem(min_idx, iter_out(min_idx, 1), lr * iter_out(min_idx, 2), lr * iter_out(min_idx, 3));
}

void Model::iter_proba(const MatrixXd& x, const VectorXd& preds, const VectorXd& y, const std::unordered_map<int, VectorXd>& qtsw)
{
    std::vector<int> indices = get_batch_indices(y.size());
    MatrixXd x_sub;
    VectorXd y_sub, preds_sub;
    const MatrixXd* x_ptr = &x;
    const VectorXd* y_ptr = &y;
    const VectorXd* preds_ptr = &preds;

    if (!indices.empty()) {
        x_sub.resize(x.rows(), indices.size());
        y_sub.resize(indices.size());
        preds_sub.resize(indices.size());
        for(size_t k = 0; k < indices.size(); ++k) {
            x_sub.col(k) = x.col(indices[k]);
            y_sub[k] = y[indices[k]];
            preds_sub[k] = preds[indices[k]];
        }
        x_ptr = &x_sub;
        y_ptr = &y_sub;
        preds_ptr = &preds_sub;
    }

    #pragma omp parallel for shared(x_ptr, preds_ptr, y_ptr, qtsw) schedule(dynamic)
    for (int i = 0; i < x_ptr->rows(); i++) {
        std::array<double, 4> out_tmp = get_best_split_proba(x_ptr->row(i), *preds_ptr, *y_ptr, qtsw.at(i));
        iter_out(i, 0) = out_tmp[0];
        iter_out(i, 1) = out_tmp[1];
        iter_out(i, 2) = out_tmp[2];
        iter_out(i, 3) = out_tmp[3];
    }
    int min_idx = 0;
    iter_out.col(0).minCoeff(&min_idx);
    VectorXd m = iter_out.row(min_idx);
    this->add_elem(min_idx, m[1], lr * m[2], lr * m[3]);
}

void Model::iter_survival(const Eigen::MatrixXd& x,
                          const Eigen::VectorXd& T,
                          const Eigen::VectorXd& E,
                          const std::unordered_map<int, Eigen::VectorXd>& qtsw,
                          const Eigen::VectorXd& f) {
    //std::array<double, 4> out_tmp;

    std::vector<int> indices = get_batch_indices(T.size());
    MatrixXd x_sub;
    VectorXd T_sub, E_sub, f_sub;
    const MatrixXd* x_ptr = &x;
    const VectorXd* T_ptr = &T;
    const VectorXd* E_ptr = &E;
    const VectorXd* f_ptr = &f;

    if (!indices.empty()) {
        x_sub.resize(x.rows(), indices.size());
        T_sub.resize(indices.size());
        E_sub.resize(indices.size());
        f_sub.resize(indices.size());
        for(size_t k = 0; k < indices.size(); ++k) {
            x_sub.col(k) = x.col(indices[k]);
            T_sub[k] = T[indices[k]];
            E_sub[k] = E[indices[k]];
            f_sub[k] = f[indices[k]];
        }
        x_ptr = &x_sub;
        T_ptr = &T_sub;
        E_ptr = &E_sub;
        f_ptr = &f_sub;
    }

    // Compute gradients for Cox loss
    //const Eigen::VectorXd gradients = compute_cox_gradients(f, T, E);
    const Eigen::VectorXd gradients = compute_ranking_gradients(*f_ptr, *T_ptr, *E_ptr, VectorXd::Ones(T_ptr->size()));
    #pragma omp parallel for shared(x_ptr, gradients, f_ptr, T_ptr, E_ptr, qtsw) default(none) schedule(dynamic)
    for (Eigen::Index i = 0; i < x_ptr->rows(); ++i) {
        // Find best split for this feature using survival objective
        std::array<double, 4> out_tmp = this->get_best_split_survival(
            x_ptr->row(i),        // current feature's values
            gradients,
            *f_ptr,
            *T_ptr,
            *E_ptr,
            qtsw.at(i)
        );

        
        //std::cout << "feat." << i << "loss: " << out_tmp[0] << "\n";
        this->iter_out(i, 0) = out_tmp[0];  // loss
        this->iter_out(i, 1) = out_tmp[1];  // split threshold
        this->iter_out(i, 2) = out_tmp[2];  // left gamma
        this->iter_out(i, 3) = out_tmp[3];  // right gamma
    }

    // Select feature with lowest loss
    Eigen::Index min_idx = 0;
    this->iter_out.col(0).minCoeff(&min_idx);
    //std::cout << "feat." << min_idx << "\n";

    // Apply best split to the model
    this->add_elem(min_idx,
                   iter_out(min_idx, 1),                   // split point
                   lr * iter_out(min_idx, 2),              // left update
                   lr * iter_out(min_idx, 3));             // right update
}

void Model::fit_proba(const MatrixXd& m, const VectorXd& y, const std::unordered_map<int, VectorXd>& qts)
{
    VectorXd res(y.size());
    res.setConstant(0.0);
    VectorXd out_model(y.size());
    for (int i = 0; i < this->max_n; i++) {
        this->iter_proba(m, res, y, qts);
        out_model = this->predict(m);
        res = out_model;
        convert_logodds_to_p_inplace(out_model);
        //std::cout("error : %f \n", cross_entropy_norm(out_model, y));
    }
    this->prune();
}

void Model::fit(const MatrixXd& m, const VectorXd& y, const std::unordered_map<int, VectorXd>& qts)
{
    VectorXd res = y; 
    VectorXd out_model(y.size());
    for (int i = 0; i < this->max_n; i++) {
        this->iter(m, res, qts);
        out_model = this->predict(m);
        res = y - out_model;
    }
    this->prune();
}

void Model::fit_survival(const MatrixXd& m,
                         const VectorXd& T,
                         const VectorXd& E,
                         const std::unordered_map<int, VectorXd>& qts) {

    VectorXd f = VectorXd::Zero(T.size());  // risk scores (log hazards)
    double loss = 0;

    for (int iter = 0; iter < this->max_n; ++iter) {
        // Each boosting step adds a tree to improve the Cox loss
        loss = ranking_loss(f,
                                   T,
                                   E,
                                   VectorXd::Ones(E.size())); 
        //std::cout << "iter: " << iter << " loss: " << loss << "\n";
        this->iter_survival(m, T, E, qts, f);
                                
        // Update prediction using the newly added tree
        f = this->predict(m);  // risk scores after adding the latest tree
    }
    this->prune();
}

void Model::set_params(const VectorXd& idxs, const VectorXd& split_val, const VectorXd& w, const double y0)
{
    this->i = idxs.size();
    // Ensure vectors are large enough? 
    // If max_n is smaller than idxs.size(), we have a problem.
    // But usually max_n is large.
    // We should resize if needed?
    if (this->idxs.size() < this->i) {
        this->idxs.resize(this->i);
        this->split_val.resize(this->i);
        this->params.w.resize(this->i);
        this->w1.resize(this->i);
        this->w2.resize(this->i);
        this->max_n = this->i;
    }
    
    this->idxs.head(this->i) = idxs;
    this->split_val.head(this->i) = split_val;
    this->params.w.head(this->i) = w;
    this->params.y0 = y0;
    
    // Zero out w1/w2 as we don't have them from JSON usually, and they are used for training mostly?
    // Or we can set them to 0.
    this->w1.setZero();
    this->w2.setZero();
}

const ScoreParams& Model::get_params() const
{
    return this->params;
}

MatrixXd Model::export_model() const
{
    MatrixXd m(4, this->idxs.size());
    m.row(0) = this->idxs;
    m.row(1) = this->split_val;
    m.row(2) = this->w1;
    m.row(3) = this->w2;
    return m;
}

std::unordered_map<int, VectorXd> make_quantiles(const MatrixXd& x, const int n_pts)
{
    std::unordered_map<int, VectorXd> l;
    for (int i = 0; i < x.rows(); i++) {
        l[i] = quantiles(x.row(i), n_pts);
    }
    return l;
}

double sum_p_complement(VectorXd& p) {
    double s = 0;
    for (int i = 0; i < p.size(); i++) {
        s += p[i] * (1 - p[i]);
    }
    return s;
}



void Model::prune() {
    std::map<std::pair<int, double>, std::tuple<double, double, double>> merged;
    
    for (int j = 0; j < this->i; j++) {
        int idx = static_cast<int>(this->idxs[j]);
        double sv = this->split_val[j];
        auto key = std::make_pair(idx, sv);
        
        if (merged.find(key) != merged.end()) {
            std::get<0>(merged[key]) += this->params.w[j];
            std::get<1>(merged[key]) += this->w1[j];
            std::get<2>(merged[key]) += this->w2[j];
        } else {
            merged[key] = std::make_tuple(this->params.w[j], this->w1[j], this->w2[j]);
        }
    }
    
    int new_size = merged.size();
    VectorXd new_idxs = VectorXd::Zero(this->max_n);
    VectorXd new_split_val = VectorXd::Zero(this->max_n);
    VectorXd new_w = VectorXd::Zero(this->max_n);
    VectorXd new_w1 = VectorXd::Zero(this->max_n);
    VectorXd new_w2 = VectorXd::Zero(this->max_n);
    
    int idx_counter = 0;
    for (const auto& [key, values] : merged) {
        new_idxs[idx_counter] = key.first;
        new_split_val[idx_counter] = key.second;
        new_w[idx_counter] = std::get<0>(values);
        new_w1[idx_counter] = std::get<1>(values);
        new_w2[idx_counter] = std::get<2>(values);
        idx_counter++;
    }
    
    this->idxs = new_idxs;
    this->split_val = new_split_val;
    this->params.w = new_w;
    this->w1 = new_w1;
    this->w2 = new_w2;
    this->i = new_size;
}
