#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> 
#include "gbrs.hpp"

namespace py = pybind11;

class PyModel {
public:
    PyModel(int n_iter, double lr, int n_quantiles, double ss_rate)
        : n_iter(n_iter), lr(lr), n_quantiles(n_quantiles), ss_rate(ss_rate) {}

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        auto qts = make_quantiles(X.transpose(), n_quantiles);
        model = std::make_unique<Model>(X.cols(), X.rows(), n_iter, lr, n_quantiles, ss_rate);
        model->fit(X.transpose(), y, qts);
    }

    void fit_proba(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        double y0 = logodds(y);
        auto qts = make_quantiles(X.transpose(), n_quantiles);
        model = std::make_unique<Model>(X.cols(), X.rows(), n_iter, lr, n_quantiles, ss_rate);
        model->params.y0 = y0;
        model->fit_proba(X.transpose(), y, qts);
    }

    void fit_survival(const Eigen::MatrixXd& X, const Eigen::VectorXd& time, const Eigen::VectorXd& event) {
        auto qts = make_quantiles(X.transpose(), n_quantiles);
        model = std::make_unique<Model>(X.cols(), X.rows(), n_iter, lr, n_quantiles, ss_rate);
        model->fit_survival(X.transpose(), time, event, qts);
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const {
        return model->predict(X.transpose());
    }

    Eigen::VectorXd predict_proba(const Eigen::MatrixXd& X) const {
        return model->predict_proba(X.transpose());
    }

    ScoreParams get_params() const {
        return model->get_params();
    }

    Eigen::VectorXd get_idxs() const {
        return model->get_idxs();
    }

    Eigen::VectorXd get_split_val() const {
        return model->get_split_val();
    }

private:
    int n_iter;
    double lr;
    int n_quantiles;
    double ss_rate;
    std::unique_ptr<Model> model;
};

PYBIND11_MODULE(core, m) {
    py::class_<ScoreParams>(m, "ScoreParams")
        .def_readwrite("y0", &ScoreParams::y0)
        .def_readwrite("w", &ScoreParams::w);

    py::class_<PyModel>(m, "Model")
        .def(py::init<int, double, int, double>(),
             py::arg("n_iter"),
             py::arg("lr"),
             py::arg("n_quantiles"),
             py::arg("ss_rate"))
        .def("fit", &PyModel::fit)
        .def("fit_proba", &PyModel::fit_proba)
        .def("fit_survival", &PyModel::fit_survival)
        .def("predict", &PyModel::predict)
        .def("predict_proba", &PyModel::predict_proba)
        .def("get_params", &PyModel::get_params)
        .def("get_idxs", &PyModel::get_idxs)
        .def("get_split_val", &PyModel::get_split_val);
}