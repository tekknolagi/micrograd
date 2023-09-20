import _imp
import argparse
import importlib
import math
import micrograd
import os
import random
import shutil
import tempfile
import time
from distutils import sysconfig
from micrograd import nn as nn_interp
from micrograd.engine import Value


random.seed(1337)
DIM = 2


def timer(lam, msg=""):
    print(msg, end=" ")
    before = time.perf_counter()
    result = lam()
    after = time.perf_counter()
    delta = after - before
    print(f"({delta:.2f} s)")
    return result


model = timer(lambda: nn_interp.MLP(DIM, [4, 1]), "Building model...")
# NOTE: It's important that input are all in sequence right next to one another
# so we can set the input in training
inp = [Value(0, (), "input") for _ in range(DIM)]
assert [i._id for i in inp] == list(range(inp[0]._id, inp[0]._id + len(inp)))
output = model(inp)
# NOTE: It's important that expected_onehot are all in sequence right next to
# one another so we can set the label in training
expected = Value(0, (), "input")
loss = (output-expected)**2
topo = timer(lambda: loss.topo(), "Building topo...")
num_nodes = len(topo)
assert num_nodes == len(set(topo)), f"{len(topo)-len(set(topo))} duplicates"
assert (
    num_nodes == micrograd.engine.counter
), f"{len(topo)} vs {micrograd.engine.counter}"


def write_code():
    with tempfile.TemporaryDirectory() as dir_path:
        source_dir = f"{dir_path}/src"
        build_dir = f"{dir_path}/build"
        os.makedirs(source_dir)
        os.makedirs(build_dir)
        file_path = f"{source_dir}/nn.c"
        with open(file_path, "w+") as f:
            print(
                f"""\
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <Python.h>
    static double data[{num_nodes}];
    static double grad[{num_nodes}];
    static inline __attribute__((always_inline)) double relu(double x) {{
    assert(isfinite(x));
        return fmax(x, 0);
    }}
    static inline __attribute__((always_inline)) double clip(double x) {{
        //assert(!isinf(x));
        //assert(!isnan(x));
        //return fmax(fmax(x, 3), -3);
    assert(isfinite(x));
        return x;
    }}
    static inline __attribute__((always_inline)) double mylog(double x) {{
    assert(isfinite(x));
        assert(x != 0);
        return log(x);
    }}
    void init() {{
    // for (int i = 0; i < {num_nodes}; i++) {{
    //     data[i] = NAN;
    // }}
            """,
                file=f,
            )
            # print("memset(data, 0, sizeof data);", file=f)
            # print("memset(grad, 0, sizeof grad);", file=f)
            print(f"for (int i = 0; i < {num_nodes}; i++) grad[i] = 0;",file=f)
            print(f"for (int i = 0; i < {num_nodes}; i++) data[i] = 0;",file=f)
            for o in model.parameters():
                print(f"data[{o._id}] = {o.data}L;", file=f)
            print("}", file=f)
            print(
                f"""\
    void set_input(PyObject* input_data) {{
        const char* buf = PyBytes_AsString(input_data);
        assert(buf);
        if (buf == NULL) {{
            abort();
        }}
        for (int i = 0; i < {DIM}; i++) {{
            data[{inp[0]._id}+i] = ((double)(unsigned char)buf[i]);
        }}
    }}
            """,
                file=f,
            )
            print("void forward() {", file=f)
            for o in topo:
                line = o.compile()
                if line:
                    print(line, file=f)
            print("}", file=f)
            print("void backward() {", file=f)
            print(f"{loss.getgrad()} = 1;", file=f)
            for o in reversed(topo):
                for line in o.backward_compile():
                    print(line, file=f)
            print("}", file=f)
            print("void update(int step, int batch_size) {", file=f)
            print("double learning_rate = 0.1;", file=f)
            for o in model.parameters():
                assert o._op in ('weight', 'bias'), repr(o._op)
                assert '[' in o.getgrad()
                print(f"data[{o._id}] -= learning_rate * {o.getgrad()} / ((double)batch_size);", file=f)
            print("}", file=f)
            print(
                f"""\
    PyObject* forward_wrapper(PyObject *module, PyObject *const *args, Py_ssize_t nargs) {{
          if (nargs != 2) {{
                PyErr_Format(PyExc_TypeError, "expected 2 args: label, pixels");
                return NULL;
          }}
          PyObject* label_obj = args[0];
          PyObject* pixels_obj = args[1];
          if (!PyLong_CheckExact(label_obj)) {{
                PyErr_Format(PyExc_TypeError, "expected int");
                return NULL;
          }}
          if (!PyBytes_CheckExact(pixels_obj)) {{
                PyErr_Format(PyExc_TypeError, "expected bytes");
                return NULL;
          }}
          if (PyBytes_Size(pixels_obj) != {DIM}) {{
                PyErr_Format(PyExc_TypeError, "expected bytes of size {DIM}");
                return NULL;
          }}
          int label = PyLong_AsLong(label_obj);
          if (label < 0) {{
                assert(PyErr_Occurred());
                return NULL;
          }}
          // Set label
          data[{expected._id}] = label;
          set_input(pixels_obj);
          forward();
          // TODO(max): Make this able to return multiple outputs?
          double loss = data[{loss._id}];
          return PyFloat_FromDouble(loss);
    }}

    PyObject* zero_grad_wrapper(PyObject* module) {{
          // Don't just zero the parameters; Karpathy can get away with that
          // because he rebuilds the entire graph every time, but we don't.
          memset(grad, 0, sizeof grad);
          // for (int i = 0; i < {num_nodes}; i++) grad[i] = 0;
          Py_RETURN_NONE;
    }}

    PyObject* backward_wrapper(PyObject* module) {{
          backward();
          Py_RETURN_NONE;
    }}

    PyObject* update_wrapper(PyObject *module, PyObject *const *args, Py_ssize_t nargs) {{
          if (nargs != 2) {{
                PyErr_Format(PyExc_TypeError, "expected 2 args: step, batch_size");
                return NULL;
          }}
          PyObject* step_obj = args[0];
          PyObject* batch_size_obj = args[1];
          int step = PyLong_AsLong(step_obj);
          if (step < 0 && PyErr_Occurred()) {{
                return NULL;
          }}
          assert(!PyErr_Occurred());
          assert(step >= 0);
          int batch_size = PyLong_AsLong(batch_size_obj);
          if (batch_size < 0 && PyErr_Occurred()) {{
                return NULL;
          }}
          assert(!PyErr_Occurred());
          assert(batch_size > 0);
          update(step, batch_size);
          Py_RETURN_NONE;
    }}

    PyObject* data_wrapper(PyObject* module, PyObject* idx_obj) {{
          long i = PyLong_AsLong(idx_obj);
          if (i < 0) {{
                  if (PyErr_Occurred()) {{
                        return NULL;
                  }}
                  PyErr_Format(PyExc_TypeError, "expected positive index");
                  return NULL;
          }}
          if (i >= {num_nodes}) {{
                  fprintf(stderr, "index %ld (dim %d)\\n", i, {num_nodes});
                  PyErr_Format(PyExc_TypeError, "index out of bounds");
                  return NULL;
          }}
          return PyFloat_FromDouble(data[i]);
    }}

    PyObject* grad_wrapper(PyObject* module, PyObject* idx_obj) {{
          long i = PyLong_AsLong(idx_obj);
          if (i < 0) {{
                  if (PyErr_Occurred()) {{
                        return NULL;
                  }}
                  PyErr_Format(PyExc_TypeError, "expected positive index");
                  return NULL;
          }}
          if (i >= {num_nodes}) {{
                  fprintf(stderr, "index %ld (dim %d)\\n", i, {num_nodes});
                  PyErr_Format(PyExc_TypeError, "index out of bounds");
                  return NULL;
          }}
          return PyFloat_FromDouble(grad[i]);
    }}

    static PyMethodDef nn_methods[] = {{
          {{ "forward", (PyCFunction)forward_wrapper, METH_FASTCALL, "doc" }},
          {{ "zero_grad", (PyCFunction)zero_grad_wrapper, METH_NOARGS, "doc" }},
          {{ "backward", (PyCFunction)backward_wrapper, METH_NOARGS, "doc" }},
          {{ "update", (PyCFunction)update_wrapper, METH_FASTCALL, "doc" }},
          {{ "data", data_wrapper, METH_O, "doc" }},
          {{ "grad", grad_wrapper, METH_O, "doc" }},
          {{ NULL, NULL }},
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
        init();
        return PyModule_Create(&nnmodule);
    }}""",
                file=f,
            )
        local_name = "nn.c"
        shutil.copyfile(f.name, local_name)
    return local_name


parser = argparse.ArgumentParser()
parser.add_argument("--use-existing", action='store_true')
args = parser.parse_args()

lib_file = "nn.so"
if not args.use_existing:
    source_file = timer(lambda: write_code(), "Writing C code...")
    # TODO(max): Bring back Extension stuff and customize compiler using
    # https://shwina.github.io/custom-compiler-linker-extensions/
    include_dir = sysconfig.get_python_inc()
    timer(
        lambda: os.system(f"tcc -g -shared -fPIC -I{include_dir} nn.c -o {lib_file}"),
        "Compiling extension...",
    )
spec = importlib.machinery.ModuleSpec("nn", None, origin=lib_file)
nn = timer(lambda: _imp.create_dynamic(spec), "Loading extension...")
print("Training...")
num_epochs = 5
db = [
    (bytes([0,0]),0),
    (bytes([0,1]),1),
    (bytes([1,0]),1),
    (bytes([1,1]),0),
]
times = 1
batch = db*times
batch_size = len(batch)
for epoch in range(num_epochs):
    epoch_loss = 0.
    nn.zero_grad()
    for im in batch:
        im_loss = nn.forward(im[1], im[0])
        out = nn.data(output._id)
        print("exp", im[1], "out", out)
        assert not math.isnan(out)
        assert not math.isinf(out)
        assert not math.isnan(im_loss)
        assert not math.isinf(im_loss)
        epoch_loss += im_loss
        nn.backward()
    nn.update(epoch, batch_size)
    epoch_loss /= batch_size
    print(f"...epoch {epoch:4d} loss {epoch_loss:.4f}")

for im in db:
    print(nn.forward(im[1], im[0]))
