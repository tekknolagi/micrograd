import _imp
import importlib
import os
import shutil
import tempfile
from distutils import sysconfig
from distutils.core import setup
from distutils.extension import Extension


def wrap_output(model):
    nout = model.nouts[-1]
    if nout == 1:
        return f"""
          double result = {model.func_name()}(input);
          return PyFloat_FromDouble(result);
        """
    return f"""
          Vector<double, {nout}> result = {model.func_name()}(input);
          PyObject *result_tuple = PyTuple_New({nout});
          for (int i = 0; i < {nout}; i++) {{
            PyTuple_SetItem(result_tuple, i, PyFloat_FromDouble(result[i]));
          }}
          return result_tuple;
    """


def make_extension(model):
    with tempfile.TemporaryDirectory() as dir_path:
        source_dir = f"{dir_path}/src"
        build_dir = f"{dir_path}/build"
        os.makedirs(source_dir)
        os.makedirs(build_dir)
        file_path = f"{source_dir}/nn.cpp"
        with open(file_path, "w+") as f:
            print(
                """\
#include <array>

//#define INLINE inline __attribute__((always_inline))
#define INLINE

    template<typename T, int dim>
    using Vector = std::array<T,dim>;
            """,
                file=f,
            )
            print("\n".join(model.compile()), file=f)
            print(
                f"""
#include <Python.h>

    extern "C" {{
    PyObject* nn_wrapper(PyObject* module, PyObject* obj) {{
          if (!PyList_CheckExact(obj)) {{
                PyErr_Format(PyExc_TypeError, "expected list");
                return nullptr;
          }}
          if (PyList_Size(obj) != {model.nin}) {{
                PyErr_Format(PyExc_TypeError, "expected list of size {model.nin}");
                return nullptr;
          }}
          Vector<double, {model.nin}> input;
          for (int i = 0; i < {model.nin}; i++) {{
            PyObject* item_obj = PyList_GetItem(obj, i);
            double item_double = PyFloat_AsDouble(item_obj);
            if (item_double < 0 && PyErr_Occurred()) {{
                return nullptr;
            }}
            input.at(i) = item_double;
          }}
          {wrap_output(model)}
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

        # TODO(max): Use shlex.split?
        print(f"C++ file is at {file_path}")
        extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
        extra_compile_args.append("-march=native")
        extra_compile_args.append("-O0")
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
        return nn_compiled
