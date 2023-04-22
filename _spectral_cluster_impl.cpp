#include <random>
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen/matrix.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Spectra/GenEigsSolver.h>

namespace py = pybind11;

using matrix_op = std::function<void(const float*, float*)>;

class NonBacktrackingMatrixOp
{
private:
    const int n;
    const matrix_op& adj_matrix_op;
    std::vector<float> deg;
public:
    using Scalar = float;

    NonBacktrackingMatrixOp(
        int n, const matrix_op& adj_matrix_op)
        : n(n), adj_matrix_op(adj_matrix_op)
    {
        deg.resize(n);
        std::vector<float> ones(n, 1.f);
        adj_matrix_op(&ones[0], &deg[0]);
        for (float& d : deg) {
            d--;
        }
    }
    int rows() const
    {
        return 2 * n;
    }
    int cols() const
    {
        return 2 * n;
    }
    void perform_op(const float* in, float* out) const
    {
        for (int i = 0; i < n; i++) {
            out[i] = -in[i + n];
        }
        adj_matrix_op(in + n, out + n);
        for (int i = 0; i < n; i++) {
            out[n + i] += deg[i] * in[i];
        }
    }
};

class cluster_error: public std::runtime_error {
    using std::runtime_error::runtime_error;
};

py::array_t<float> get_eigenvectors(int n, const matrix_op& adj_matrix_op,
    int count, int max_iter, float tol, std::optional<int> ncv, bool debug)
{
    NonBacktrackingMatrixOp op(n, adj_matrix_op);
    Spectra::GenEigsSolver<NonBacktrackingMatrixOp> solver(op, count + 1, ncv.value_or(2 * count + 2));
    std::vector<float> initial_vec(2 * n, 1);
    solver.init(initial_vec.data());
    if (debug) {
        std::cout << "Begin solver.compute()" << std::endl;
    }
    solver.compute(Spectra::SortRule::LargestReal, max_iter, tol, Spectra::SortRule::LargestReal);
    if (debug) {
        std::cout << "End solver.compute()" << std::endl;
    }
    if (solver.info() != Spectra::CompInfo::Successful) {
        throw cluster_error("Solver failure");
    }
    if (debug) {
        std::cout << "#iter = " << solver.num_iterations() << ", #mul = " << solver.num_operations() << std::endl;
    }
    auto evals = solver.eigenvalues();
    auto evecs = solver.eigenvectors();
    for (int j = 0; j <= count; j++) {
        if (std::abs(evals[j].imag()) > 1e-6) {
            throw cluster_error("Non-real eigenvalue");
        }
    }
    py::array_t<float> result({ n, count });
    auto ptr = result.mutable_unchecked<2>();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < count; j++) {
            ptr(i, j) = evecs(i, j + 1).real();
        }
    }
    return result;
}

py::array_t<float> get_eigenvectors_from_op(int n,
    std::function<void(py::array_t<float>, py::array_t<float>)> adj_matrix_op,
    int count, int max_iter, float tol, std::optional<int> ncv, bool debug)
{
    return get_eigenvectors(n,
        [n, &adj_matrix_op](const float* in, float* out) {
            {
                py::capsule c1([]() {});
                py::capsule c2([]() {});
                adj_matrix_op(
                    py::array_t<float>({ n }, { sizeof(float) }, in, c1),
                    py::array_t<float>({ n }, { sizeof(float) }, out, c2));
            }
        },
        count, max_iter, tol, ncv, debug);
}

template<class Matrix>
py::array_t<float> get_eigenvectors_from_matrix(const Matrix& adj_matrix,
    int count, int max_iter, float tol, std::optional<int> ncv, bool debug)
{
    if (adj_matrix.rows() != adj_matrix.cols()) {
        throw std::invalid_argument("adj_matrix must be a square matrix");
    }
    int n = adj_matrix.rows();
    matrix_op adj_matrix_op([n, &adj_matrix](const float* in, float* out) {
        Eigen::Map<const Eigen::Vector<float, Eigen::Dynamic>> in_vec(in, n);
        Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>> out_vec(out, n);
        out_vec = adj_matrix.template cast<float>() * in_vec;
        });
    return get_eigenvectors(n, adj_matrix_op, count, max_iter, tol, ncv, debug);
}

std::pair<Eigen::VectorXi, Eigen::SparseMatrix<bool, Eigen::RowMajor>> generate_sparse_sbm(
    const Eigen::VectorXi& sizes, const Eigen::MatrixXf& prob_matrix, std::optional<unsigned int> seed)
{
    if (sizes.rows() != prob_matrix.rows() || sizes.rows() != prob_matrix.cols()) {
        throw std::invalid_argument("Invalid input size");
    }
    if (!seed.has_value()) {
        std::random_device dev;
        seed = dev();
    }
    std::mt19937 engine(seed.value());
    int n = sizes.sum();
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), engine);
    Eigen::VectorXi labels(n);
    std::vector<Eigen::Triplet<bool>> edges;
    std::vector<int> deg(n);
    Eigen::SparseMatrix<bool, Eigen::RowMajor> adj(n, n);
    int pos_i = 0;
    for (int i = 0; i < sizes.rows(); i++) {
        for (int v = pos_i; v < pos_i + sizes[i]; v++) {
            int pos_j = pos_i;
            for (int j = i; j < sizes.rows(); j++) {
                std::geometric_distribution<int> dist(prob_matrix(i, j));
                int last = std::max(pos_j, v + 1);
                while (true) {
                    int u = last + dist(engine);
                    if (u >= pos_j + sizes[j]) {
                        break;
                    }
                    last = u + 1;
                    edges.emplace_back(perm[v], perm[u], true);
                    edges.emplace_back(perm[u], perm[v], true);
                }
                pos_j += sizes[j];
            }
            labels[perm[v]] = i;
        }
        pos_i += sizes[i];
    }
    adj.setFromTriplets(edges.begin(), edges.end());
    return std::make_pair(labels, adj);
}

PYBIND11_MODULE(_spectral_cluster_impl, m)
{
    m.def("get_eigenvectors_from_op", &get_eigenvectors_from_op, py::arg("n"), py::arg("adj_matrix_op"), py::arg("count"),
        py::kw_only(), py::arg("max_iter"), py::arg("tol"), py::arg("ncv"), py::arg("debug"));
    m.def("get_eigenvectors_from_sparse",
        &get_eigenvectors_from_matrix<Eigen::SparseMatrix<bool, Eigen::RowMajor>>,
        py::arg("adj_matrix"), py::arg("count"),
        py::kw_only(), py::arg("max_iter"), py::arg("tol"), py::arg("ncv"), py::arg("debug"));
    m.def("get_eigenvectors_from_dense",
        &get_eigenvectors_from_matrix<Eigen::MatrixX<bool>>,
        py::arg("adj_matrix"), py::arg("count"),
        py::kw_only(), py::arg("max_iter"), py::arg("tol"), py::arg("ncv"), py::arg("debug"));
    m.def("generate_sparse_sbm", &generate_sparse_sbm, py::arg("sizes"), py::arg("prob_matrix"), py::arg("seed"));
    py::register_exception<cluster_error>(m, "ClusterError");
}