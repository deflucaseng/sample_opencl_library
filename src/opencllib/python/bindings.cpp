#include <Python.h>
#include "structmember.h"
#include "../cpp/matrix.hpp"
#include "../cpp/operationmanager.hpp"


// Matrix wrapper
typedef struct {
    PyObject_HEAD
    Matrix* matrix;
} PyMatrix;

static void PyMatrix_dealloc(PyMatrix* self) {
    delete self->matrix;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyMatrix_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyMatrix* self = (PyMatrix*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static int PyMatrix_init(PyMatrix* self, PyObject* args, PyObject* kwds) {
    PyObject* data_obj;
    int height, width;
    
    if (!PyArg_ParseTuple(args, "Oii", &data_obj, &height, &width))
        return -1;
    
    if (!PyList_Check(data_obj)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be a list");
        return -1;
    }
    
    Py_ssize_t size = PyList_Size(data_obj);
    if (size != height * width) {
        PyErr_SetString(PyExc_ValueError, "Data size doesn't match dimensions");
        return -1;
    }
    
    float* data = new float[size];
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyList_GetItem(data_obj, i);
        data[i] = (float)PyFloat_AsDouble(item);
    }
    
    self->matrix = new Matrix(data, height, width);
    return 0;
}

static PyObject* PyMatrix_get_data(PyMatrix* self, PyObject* Py_UNUSED(ignored)) {
    float* data = self->matrix->get_data();
    int size = self->matrix->get_height() * self->matrix->get_width();
    
    PyObject* list = PyList_New(size);
    for (int i = 0; i < size; i++) {
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(data[i]));
    }
    
    return list;
}

static PyMethodDef PyMatrix_methods[] = {
    {"get_data", (PyCFunction)PyMatrix_get_data, METH_NOARGS, "Get matrix data"},
    {NULL}
};

static PyTypeObject PyMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "matrix.Matrix",
    .tp_basicsize = sizeof(PyMatrix),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)PyMatrix_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Matrix object",
    .tp_methods = PyMatrix_methods,
    .tp_init = (initproc)PyMatrix_init,
    .tp_new = PyMatrix_new
};

// OperationManager wrapper
typedef struct {
    PyObject_HEAD
    OperationManager* op_manager;
} PyOperationManager;

static void PyOperationManager_dealloc(PyOperationManager* self) {
    delete self->op_manager;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyOperationManager_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyOperationManager* self = (PyOperationManager*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static int PyOperationManager_init(PyOperationManager* self, PyObject* args, PyObject* kwds) {
    const char* device_type_str;
    if (!PyArg_ParseTuple(args, "s", &device_type_str))
        return -1;
    
    OperationManager::DeviceType device_type;
    if (strcmp(device_type_str, "CPU") == 0)
        device_type = OperationManager::DeviceType::CPU_DEVICE;
    else if (strcmp(device_type_str, "GPU") == 0)
        device_type = OperationManager::DeviceType::GPU_DEVICE;
    else {
        PyErr_SetString(PyExc_ValueError, "Device type must be 'CPU' or 'GPU'");
        return -1;
    }
    
    self->op_manager = new OperationManager(device_type);
    return 0;
}

static PyObject* PyOperationManager_element_wise_operation(PyOperationManager* self, PyObject* args) {
    const char* op_type_str;
    PyMatrix* lhs;
    PyMatrix* rhs;
    
    if (!PyArg_ParseTuple(args, "sO!O!", &op_type_str, 
                         &PyMatrixType, &lhs, &PyMatrixType, &rhs))
        return NULL;
        
    operation_types op_type;
    if (strcmp(op_type_str, "add") == 0)
        op_type = operation_types::ADD;
    else if (strcmp(op_type_str, "subtract") == 0)
        op_type = operation_types::SUB;
    else if (strcmp(op_type_str, "multiply") == 0)
        op_type = operation_types::MUL;
    else if (strcmp(op_type_str, "divide") == 0)
        op_type = operation_types::DIV;
    else {
        PyErr_SetString(PyExc_ValueError, "Invalid operation type");
        return NULL;
    }
    
    float* result = self->op_manager->element_wise_operation(
        op_type, lhs->matrix->get_data(), rhs->matrix->get_data(),
        lhs->matrix->get_height(), lhs->matrix->get_width());
        
    // Convert result back to Python list
    PyObject* result_list = PyList_New(lhs->matrix->get_height() * lhs->matrix->get_width());
    for (int i = 0; i < lhs->matrix->get_height() * lhs->matrix->get_width(); i++) {
        PyList_SET_ITEM(result_list, i, PyFloat_FromDouble(result[i]));
    }
    
    delete[] result;
    return result_list;
}

static PyMethodDef PyOperationManager_methods[] = {
    {"element_wise_operation", (PyCFunction)PyOperationManager_element_wise_operation, 
     METH_VARARGS, "Perform element-wise operation"},
    {NULL}
};

static PyTypeObject PyOperationManagerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "matrix.OperationManager",
    .tp_basicsize = sizeof(PyOperationManager),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)PyOperationManager_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "OperationManager object",
    .tp_methods = PyOperationManager_methods,
    .tp_init = (initproc)PyOperationManager_init,
    .tp_new = PyOperationManager_new
};

// Module initialization
static PyModuleDef matrixmodule = {
    PyModuleDef_HEAD_INIT,
    "sample_opencl_lib",  // Changed from "matrix"
    "Matrix operations module",
    -1,
    NULL
};

PyMODINIT_FUNC PyInit_sample_opencl_lib(void) {  // Changed from PyInit_matrix
    PyObject* m;
    
    if (PyType_Ready(&PyMatrixType) < 0)
        return NULL;
    if (PyType_Ready(&PyOperationManagerType) < 0)
        return NULL;
        
    m = PyModule_Create(&matrixmodule);
    if (m == NULL)
        return NULL;
        
    Py_INCREF(&PyMatrixType);
    Py_INCREF(&PyOperationManagerType);
    PyModule_AddObject(m, "Matrix", (PyObject*)&PyMatrixType);
    PyModule_AddObject(m, "OperationManager", (PyObject*)&PyOperationManagerType);
    
    return m;
}