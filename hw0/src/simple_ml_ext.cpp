#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (size_t b = 0; b < m; b += batch) {
        // for each batch:

        // calculate Z - I_y (after normalize)
        std::vector<std::vector<float>> Z(batch, std::vector<float>(k, 0));
        //float Z[batch][k] = {};
        size_t z_row, z_col;
        for (z_row = 0; z_row < batch; z_row++) {
            float sum = 0; // row-wise sum
            for (z_col = 0; z_col < k; z_col++) {
                float z_element = 0;
                for (size_t i = 0; i < n; i++) {
                    z_element += X[(b+z_row)*n + i] * theta[i*k + z_col];
                }
                z_element = std::exp(z_element);
                Z[z_row][z_col] = z_element;
                sum += z_element;
            }
            
            // normalize and minus e_y
            for (z_col = 0; z_col < k; z_col++) {
                Z[z_row][z_col] /= sum;
                if (z_col == y[b+z_row])
                    Z[z_row][z_col] -= 1;
            }
        }

        // calculate grad: X.T @ (Z - I_y) / batch
        // and update in place
        size_t g_row, g_col;
        for (g_row = 0; g_row < n; g_row++) {
            for (g_col = 0; g_col < k; g_col++) {
                for (size_t i = 0; i < batch; i++) {
                    theta[g_row*k + g_col] -= X[(b+i)*n + g_row] * Z[i][g_col] * lr / batch;
                }
                
            }
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
