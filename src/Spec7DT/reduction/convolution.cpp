#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <fftw3.h>
#include <math.h>
#include <string.h> // For memset

// Forward declaration
static PyObject* convolve_fft_c(PyObject *self, PyObject *args);

static PyObject* convolve_fft_c(PyObject *self, PyObject *args) {
    PyObject *image_obj = NULL;
    PyObject *kernel_obj = NULL;
    PyArrayObject *image_arr = NULL;
    PyArrayObject *kernel_arr = NULL;
    PyArrayObject *result_arr = NULL;
    PyObject *result_obj = NULL;

    // FFTW pointers
    float *image_clean_ptr = NULL;
    float *weights_ptr = NULL;
    float *padded_kernel = NULL;
    float *convolved_image = NULL;
    float *convolved_weights = NULL;
    fftwf_complex *image_fft = NULL;
    fftwf_complex *kernel_fft = NULL;
    fftwf_complex *weights_fft = NULL;
    fftwf_complex *conv_fft = NULL;
    fftwf_plan p_image_fwd = NULL;
    fftwf_plan p_weights_fwd = NULL;
    fftwf_plan p_kernel_fwd = NULL;
    fftwf_plan p_image_bwd = NULL;
    fftwf_plan p_weights_bwd = NULL;

    if (!PyArg_ParseTuple(args, "OO", &image_obj, &kernel_obj)) {
        return NULL;
    }

    // Interpret the input objects as NumPy arrays
    image_arr = (PyArrayObject*)PyArray_FROM_OTF(image_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    kernel_arr = (PyArrayObject*)PyArray_FROM_OTF(kernel_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    if (image_arr == NULL || kernel_arr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to convert inputs to NumPy arrays.");
        goto cleanup;
    }

    if (PyArray_NDIM(image_arr) != 2 || PyArray_NDIM(kernel_arr) != 2) {
        PyErr_SetString(PyExc_RuntimeError, "Input should be 2-D NumPy arrays.");
        goto cleanup;
    }

    // Get dimensions
    npy_intp *image_dims = PyArray_DIMS(image_arr);
    npy_intp *kernel_dims = PyArray_DIMS(kernel_arr);
    long rows = image_dims[0];
    long cols = image_dims[1];
    long k_rows = kernel_dims[0];
    long k_cols = kernel_dims[1];
    long n_cells = rows * cols;
    long fft_cols = cols / 2 + 1;
    long n_complex = rows * fft_cols;

    // Get data pointers
    float *image_ptr = (float *)PyArray_DATA(image_arr);
    float *kernel_ptr = (float *)PyArray_DATA(kernel_arr);

    // Allocate memory
    image_clean_ptr = (float*) fftwf_malloc(sizeof(float) * n_cells);
    weights_ptr = (float*) fftwf_malloc(sizeof(float) * n_cells);
    padded_kernel = (float*) fftwf_malloc(sizeof(float) * n_cells);
    convolved_image = (float*) fftwf_malloc(sizeof(float) * n_cells);
    convolved_weights = (float*) fftwf_malloc(sizeof(float) * n_cells);
    image_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_complex);
    kernel_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_complex);
    weights_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_complex);
    conv_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n_complex);

    if (!image_clean_ptr || !weights_ptr || !padded_kernel || !convolved_image || !convolved_weights ||
        !image_fft || !kernel_fft || !weights_fft || !conv_fft) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for FFTW arrays.");
        goto cleanup;
    }

    // --- Handle NaNs with normalized convolution ---
    for (long i = 0; i < n_cells; ++i) {
        if (isnan(image_ptr[i])) {
            image_clean_ptr[i] = 0.0f;
            weights_ptr[i] = 0.0f;
        } else {
            image_clean_ptr[i] = image_ptr[i];
            weights_ptr[i] = 1.0f;
        }
    }

    // --- Prepare padded and shifted kernel ---
    memset(padded_kernel, 0, sizeof(float) * n_cells);
    for (long r = 0; r < k_rows; ++r) {
        for (long c = 0; c < k_cols; ++c) {
            long pr = (r - k_rows / 2 + rows) % rows;
            long pc = (c - k_cols / 2 + cols) % cols;
            padded_kernel[pr * cols + pc] = kernel_ptr[r * k_cols + c];
        }
    }

    // Create FFTW plans
    p_image_fwd = fftwf_plan_dft_r2c_2d(rows, cols, image_clean_ptr, image_fft, FFTW_ESTIMATE);
    p_weights_fwd = fftwf_plan_dft_r2c_2d(rows, cols, weights_ptr, weights_fft, FFTW_ESTIMATE);
    p_kernel_fwd = fftwf_plan_dft_r2c_2d(rows, cols, padded_kernel, kernel_fft, FFTW_ESTIMATE);
    p_image_bwd = fftwf_plan_dft_c2r_2d(rows, cols, conv_fft, convolved_image, FFTW_ESTIMATE);
    p_weights_bwd = fftwf_plan_dft_c2r_2d(rows, cols, conv_fft, convolved_weights, FFTW_ESTIMATE);

    // --- Execute FFTs ---
    fftwf_execute(p_image_fwd);
    fftwf_execute(p_weights_fwd);
    fftwf_execute(p_kernel_fwd);

    // --- Point-wise multiplication in frequency domain for image ---
    for (long i = 0; i < n_complex; ++i) {
        conv_fft[i][0] = image_fft[i][0] * kernel_fft[i][0] - image_fft[i][1] * kernel_fft[i][1];
        conv_fft[i][1] = image_fft[i][0] * kernel_fft[i][1] + image_fft[i][1] * kernel_fft[i][0];
    }
    fftwf_execute(p_image_bwd);

    // --- Point-wise multiplication in frequency domain for weights ---
    for (long i = 0; i < n_complex; ++i) {
        conv_fft[i][0] = weights_fft[i][0] * kernel_fft[i][0] - weights_fft[i][1] * kernel_fft[i][1];
        conv_fft[i][1] = weights_fft[i][0] * kernel_fft[i][1] + weights_fft[i][1] * kernel_fft[i][0];
    }
    fftwf_execute(p_weights_bwd);

    // --- Create result array and normalize ---
    npy_intp dims[] = {rows, cols};
    result_arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (result_arr == NULL) {
        goto cleanup;
    }
    float *result_ptr = (float *)PyArray_DATA(result_arr);
    float norm_factor = 1.0f / n_cells;

    for (long i = 0; i < n_cells; ++i) {
        if (convolved_weights[i] > 1e-9f) {
            result_ptr[i] = (convolved_image[i] / convolved_weights[i]) * norm_factor;
        } else {
            result_ptr[i] = 0.0f;
        }
    }
    
    result_obj = (PyObject*)result_arr;

cleanup:
    if(p_image_fwd) fftwf_destroy_plan(p_image_fwd);
    if(p_weights_fwd) fftwf_destroy_plan(p_weights_fwd);
    if(p_kernel_fwd) fftwf_destroy_plan(p_kernel_fwd);
    if(p_image_bwd) fftwf_destroy_plan(p_image_bwd);
    if(p_weights_bwd) fftwf_destroy_plan(p_weights_bwd);

    if(image_clean_ptr) fftwf_free(image_clean_ptr);
    if(weights_ptr) fftwf_free(weights_ptr);
    if(padded_kernel) fftwf_free(padded_kernel);
    if(convolved_image) fftwf_free(convolved_image);
    if(convolved_weights) fftwf_free(convolved_weights);
    if(image_fft) fftwf_free(image_fft);
    if(kernel_fft) fftwf_free(kernel_fft);
    if(weights_fft) fftwf_free(weights_fft);
    if(conv_fft) fftwf_free(conv_fft);

    Py_XDECREF(image_arr);
    Py_XDECREF(kernel_arr);
    
    if (result_obj == NULL) {
        Py_XDECREF(result_arr);
    }

    return result_obj;
}

// Method definition table
static PyMethodDef ConvolutionMethods[] = {
    {"convolve_fft_c",  convolve_fft_c, METH_VARARGS,
     "A function which convolves two 2D numpy arrays using FFTW (C implementation), handling NaNs with normalization."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

// Module definition structure
static struct PyModuleDef convolutionmodule = {
    PyModuleDef_HEAD_INIT,
    "convolution",   /* name of module */
    "C implementation of FFT convolution.", /* module documentation, may be NULL */
    -1,
    ConvolutionMethods
};

// Module initialization function
PyMODINIT_FUNC
PyInit_convolution(void)
{
    PyObject *m = PyModule_Create(&convolutionmodule);
    if (m == NULL) {
        return NULL;
    }
    import_array(); // Initialize NumPy C-API
    if (PyErr_Occurred()) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}