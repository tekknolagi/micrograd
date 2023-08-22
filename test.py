import random
import tempfile
import os
import importlib
import _imp
import shutil
from micrograd import nn as nn_interp
from micrograd.engine import Value

random.seed(1337)
n = nn_interp.MLP(2, [16, 16, 1])
x = [Value(i) for i in range(2)]
expected = n(x)

with tempfile.TemporaryDirectory() as dir_path:
    source_dir = f"{dir_path}/src"
    build_dir = f"{dir_path}/build"
    os.makedirs(source_dir)
    os.makedirs(build_dir)
    file_path = f"{source_dir}/nn.cpp"
    with open(file_path, "w+") as f:
        print(f"// {x} -> {expected}", file=f)
        print(
            """\
#include <cstring>
#include <initializer_list>
#include <algorithm>
#include <cassert>
#include <array>

#define INLINE inline __attribute__((always_inline))

template <typename T = double, int dim = 1>
class Vector {
 public:
  INLINE Vector<T, dim>() { arr.fill(0); }
  INLINE Vector<T, dim>(T other[dim]) {
    for (int i = 0; i < dim; i++) {
      arr[i] = other[i];
    }
  }
  INLINE Vector<T, dim>(std::initializer_list<T> other) {
    assert(other.size() == dim && "oh no");
    for (int i = 0; i < dim; i++) {
      arr[i] = other.begin()[i];
    }
  }
  INLINE T sum() const {
    T result = 0;
    for (int i = 0; i < dim; i++) {
      result += arr[i];
    }
    return result;
  }
  INLINE T& at(int idx) { return arr[idx]; }
  INLINE const T& at(int idx) const { return arr[idx]; }

 private:
  std::array<T, dim> arr;
};
        """,
            file=f,
        )
        print("\n".join(n.compile()), file=f)
        print(
            f"""
#include <Python.h>

extern "C" {{
PyObject* nn_wrapper(PyObject* module, PyObject* obj) {{
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
        PyObject* item_obj = PyList_GetItem(obj, i);
        double item_double = PyFloat_AsDouble(item_obj);
        if (item_double < 0 && PyErr_Occurred()) {{
            return nullptr;
        }}
        input.at(i) = item_double;
      }}
      // TODO(max): Make this able to return multiple outputs?
      double result = {n.func_name()}(input);
      return PyFloat_FromDouble(result);
}}

static PyMethodDef nn_methods[] = {{
      {{ "nn", nn_wrapper, METH_O, "doc" }},
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
        """,
            file=f,
        )
    from distutils.core import setup
    from distutils.extension import Extension
    from distutils import sysconfig

    # TODO(max): Use shlex.split?
    extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
    extra_compile_args.append("-march=native")
    ext = Extension(name="nn", sources=[file_path], extra_compile_args=extra_compile_args)
    setup(
        name="nn",
        ext_modules=[ext],
        script_args=["--quiet", "build_ext", "--build-temp", build_dir],
        options={"build_ext": {"build_lib": source_dir}},
    )
    dir_contents = sorted(os.listdir(source_dir))
    assert dir_contents[1].endswith(".so")
    lib_file = f"{source_dir}/{dir_contents[1]}"
    spec = importlib.machinery.ModuleSpec("nn", None, origin=lib_file)
    nn_compiled = _imp.create_dynamic(spec)
    actual = nn_compiled.nn([xi.data for xi in x])
    shutil.copyfile(f.name, "nn.cpp")
    shutil.copyfile(lib_file, "nn.so")
    # assert expected.data == actual, f"expected {expected} but got {actual} (diff {expected.data-actual})"
    print(f"Karpathy's micrograd produces : {expected.data}")
    print(f"Bernstein's micrograd produces: {actual}")
