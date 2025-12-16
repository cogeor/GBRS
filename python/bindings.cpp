#include "lib.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Helper function to build quantile map from user-provided or auto-generated
// quantiles
std::unordered_map<int, Eigen::VectorXd>
build_quantile_map(const Eigen::MatrixXd &m, int n_quantiles,
                   py::object user_quantiles) {
  int n_features = m.rows();
  std::unordered_map<int, Eigen::VectorXd> qts;

  if (!user_quantiles.is_none()) {
    // user_quantiles should be a list of numpy arrays
    py::list qlist = user_quantiles.cast<py::list>();

    if (qlist.size() != static_cast<size_t>(n_features)) {
      throw std::runtime_error("Length of user_quantiles list must match "
                               "number of features (columns in X)");
    }

    for (int i = 0; i < n_features; ++i) {
      if (!qlist[i].is_none()) {
        // Convert numpy array to Eigen vector
        Eigen::VectorXd q = qlist[i].cast<Eigen::VectorXd>();
        qts[i] = q;
      } else {
        // Use auto-generated quantiles for this feature
        Eigen::VectorXd col = m.row(i);
        qts[i] = quantiles(col, n_quantiles);
      }
    }
  } else {
    // Use auto-generated quantiles for all features
    qts = make_quantiles(m, n_quantiles);
  }

  return qts;
}

class PyModel {
public:
  PyModel(int n_iter, double lr, int n_quantiles, int batch_size)
      : n_iter(n_iter), lr(lr), n_quantiles(n_quantiles),
        batch_size(batch_size) {}

  void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
           py::object user_quantiles = py::none()) {
    auto qts = build_quantile_map(X.transpose(), n_quantiles, user_quantiles);
    model = std::make_unique<Model>(X.cols(), X.rows(), n_iter, lr, n_quantiles,
                                    batch_size);
    model->fit(X.transpose(), y, qts);
  }

  void fit_proba(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                 py::object user_quantiles = py::none()) {
    double y0 = logodds(y);
    auto qts = build_quantile_map(X.transpose(), n_quantiles, user_quantiles);
    model = std::make_unique<Model>(X.cols(), X.rows(), n_iter, lr, n_quantiles,
                                    batch_size);
    model->params.y0 = y0;
    model->fit_proba(X.transpose(), y, qts);
  }

  void fit_survival(const Eigen::MatrixXd &X, const Eigen::VectorXd &time,
                    const Eigen::VectorXd &event,
                    py::object user_quantiles = py::none()) {
    auto qts = build_quantile_map(X.transpose(), n_quantiles, user_quantiles);
    model = std::make_unique<Model>(X.cols(), X.rows(), n_iter, lr, n_quantiles,
                                    batch_size);
    model->fit_survival(X.transpose(), time, event, qts);
  }

  Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
    return model->predict(X.transpose());
  }

  Eigen::VectorXd predict_proba(const Eigen::MatrixXd &X) const {
    return model->predict_proba(X.transpose());
  }

  ScoreParams get_params() const { return model->get_params(); }

  Eigen::VectorXd get_idxs() const { return model->get_idxs(); }

  Eigen::VectorXd get_split_val() const { return model->get_split_val(); }

  void set_params(const Eigen::VectorXd &idxs, const Eigen::VectorXd &split_val,
                  const Eigen::VectorXd &w, double y0) {
    if (!model) {
      model = std::make_unique<Model>(0, 0, idxs.size(), 0.0, 0, 0.0);
    }
    model->set_params(idxs, split_val, w, y0);
  }

private:
  int n_iter;
  double lr;
  int n_quantiles;
  int batch_size;
  std::unique_ptr<Model> model;
};

PYBIND11_MODULE(core, m) {
  py::class_<ScoreParams>(m, "ScoreParams")
      .def_readwrite("y0", &ScoreParams::y0)
      .def_readwrite("w", &ScoreParams::w);

  py::class_<PyModel>(m, "Model")
      .def(py::init<int, double, int, int>(), py::arg("n_iter"), py::arg("lr"),
           py::arg("n_quantiles"), py::arg("batch_size"))
      .def("fit", &PyModel::fit, py::arg("X"), py::arg("y"),
           py::arg("user_quantiles") = py::none())
      .def("fit_proba", &PyModel::fit_proba, py::arg("X"), py::arg("y"),
           py::arg("user_quantiles") = py::none())
      .def("fit_survival", &PyModel::fit_survival, py::arg("X"),
           py::arg("time"), py::arg("event"),
           py::arg("user_quantiles") = py::none())
      .def("predict", &PyModel::predict)
      .def("predict_proba", &PyModel::predict_proba)
      .def("get_params", &PyModel::get_params)
      .def("get_idxs", &PyModel::get_idxs)
      .def("get_split_val", &PyModel::get_split_val)
      .def("set_params", &PyModel::set_params);
}