import random
from micrograd import nn
from micrograd.engine import Value

random.seed(1337)
n = nn.MLP(2, [16, 16, 1])
x = [Value(i) for i in range(2)]
print("//", n(x))
print("""\
#include <cstring>
#include <initializer_list>
#include <algorithm>
#include <cassert>

template <typename T = double, int dim = 1>
class Vector {
 public:
  Vector<T, dim>() { std::memset(arr, 0, dim*sizeof(T)); }
  Vector<T, dim>(T other[dim]) {
    for (int i = 0; i < dim; i++) {
      arr[i] = other[i];
    }
  }
  Vector<T, dim>(std::initializer_list<T> other) {
    assert(other.size() == dim && "oh no");
    for (int i = 0; i < dim; i++) {
      arr[i] = other.begin()[i];
    }
  }
  Vector<T, dim> dot(Vector<T, dim> other) {
    T result[dim];
    for (int i = 0; i < dim; i++) {
      arr[i] = other.arr[i];
    }
    return result;
  }
  T sum() {
    T result = 0;
    for (int i = 0; i < dim; i++) {
      result += arr[i];
    }
    return result;
  }
  T& at(int idx) { return arr[idx]; }

 private:
  T arr[dim];
};
""")
print("\n".join(n.compile()))
print(f"""
#include <Python.h>

extern "C" {{
PyObject* nn_wrapper(PyObject* obj) {{
      if (!PyList_CheckExact(obj)) {{
            PyErr_Format(PyExc_TypeError, "expected list");
            return nullptr;
      }}
      if (PyList_Size(obj) != {n.nin}) {{
            PyErr_Format(PyExc_TypeError, "expected list of size {n.nin}");
            return nullptr;
      }}
      Vector<double, {n.nin}> input;
      for (int i = 0; i < {n.nin}; i++) {{
        inputs.at(i) = PyList_GetItem(obj, i);
      }}
      // TODO(max): Make this able to return multiple outputs?
      double result = {n.func_name()}(input);
      return PyFloat_FromDouble(result);
}}

static PyMethodDef nn_methods[] = {{
      {{ "nn", nn_wrapper, METH_OBJECT, "doc" }},
      {{ nullptr, nullptr }},
}};

// clang-format off
static struct PyModuleDef nnmodule = {{
    PyModuleDef_HEAD_INIT,
    "nn",
    "doc",
    -1,
    nn_methods,
    NULL,
    NULL,
    NULL,
    NULL
}};
// clang-format on

PyObject* PyInit_nn() {{
    PyObject* m = PyState_FindModule(&nnmodule);
    if (m != NULL) {{
        return m;
    }}
    return PyModule_Create(&nnmodule);
}}
}}
""")
