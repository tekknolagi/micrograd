import argparse
import _imp
import collections
import importlib
import math
import micrograd
import os
import random
import shutil
import struct
import tempfile
import time
from distutils import sysconfig
from micrograd import nn as nn_interp
from micrograd.engine import Value


random.seed(1337)


class image:
    def __init__(self, label, pixels):
        self.label = label
        self.pixels = pixels


IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
PIXEL_LENGTH = IMAGE_HEIGHT * IMAGE_WIDTH
DIM = PIXEL_LENGTH


class images:
    def __init__(self, images_filename, labels_filename):
        self.images = open(images_filename, "rb")
        self.labels = open(labels_filename, "rb")
        self.idx = 0
        self.read_magic()

    def read_magic(self):
        images_magic = self.images.read(4)
        assert images_magic == b"\x00\x00\x08\x03"
        labels_magic = self.labels.read(4)
        assert labels_magic == b"\x00\x00\x08\x01"
        (self.num_images,) = struct.unpack(">L", self.images.read(4))
        (self.num_labels,) = struct.unpack(">L", self.labels.read(4))
        assert self.num_images == self.num_labels
        nrows = self.images.read(4)
        assert struct.unpack(">L", nrows) == (IMAGE_HEIGHT,)
        ncols = self.images.read(4)
        assert struct.unpack(">L", ncols) == (IMAGE_WIDTH,)

    def read_image(self):
        label = int.from_bytes(self.labels.read(1), "big")
        pixels = self.images.read(PIXEL_LENGTH)
        self.idx += 1
        return image(label, pixels)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_images:
            raise StopIteration
        return self.read_image()

    def num_left(self):
        return self.num_images - self.idx


def timer(lam, msg=""):
    print(msg, end=" ")
    before = time.perf_counter()
    result = lam()
    after = time.perf_counter()
    delta = after - before
    print(f"({delta:.2f} s)")
    return result


db = timer(
    lambda: images("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    "Opening images...",
)
NUM_DIGITS = 10
model = timer(lambda: nn_interp.MLP(DIM, [512, NUM_DIGITS]), "Building model...")
# NOTE: It's important that input are all in sequence right next to one another
# so we can set the input in training
inp = [Value(0, (), "input") for _ in range(DIM)]
assert [i._id for i in inp] == list(range(inp[0]._id, inp[0]._id + len(inp)))
out = model(inp)
# NOTE: It's important that expected_onehot are all in sequence right next to
# one another so we can set the label in training
expected_onehot = [Value(0, (), "input") for _ in range(NUM_DIGITS)]
assert [exp._id for exp in expected_onehot] == list(
    range(expected_onehot[0]._id, expected_onehot[0]._id + len(expected_onehot))
)
loss = sum((exp - act) ** 2 for exp, act in zip(expected_onehot, out))
topo = timer(lambda: loss.topo(), "Building topo...")
# TODO(max): Figure out why there are (significant numbers of) duplicated
# Values
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
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <Python.h>
    double data[{num_nodes}];
    double grad[{num_nodes}];
    double relu(double x) {{ if (x < 0) {{ return 0; }} else {{ return x; }} }}
    void init() {{
            """,
                file=f,
            )
            for o in model.parameters():
                print(f"data[{o._id}] = {o.data}L;", file=f)
            print("}", file=f)
            print(
                f"""\
    void set_input(PyObject* input_data) {{
        const char* buf = PyBytes_AsString(input_data);
        if (buf == NULL) {{
            abort();
        }}
        for (int i = 0; i < {DIM}; i++) {{
            data[{inp[0]._id}+i] = buf[i];
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
            print(f"grad[{loss._id}] = 1;", file=f)
            for o in reversed(topo):
                for line in o.backward_compile():
                    print(line, file=f)
            print("}", file=f)
            print("void update(int step) {", file=f)
            # TODO(max): It's not always 100; is this hard-coded for number of
            # training rounds in Karpathy's code?
            print(
                "double learning_rate = 1.0L - (0.9L * (double)step) / 100.0L;", file=f
            )
            for o in model.parameters():
                print(f"data[{o._id}] -= learning_rate * grad[{o._id}];", file=f)
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
          if (label < 0 && PyErr_Occurred()) {{
                return NULL;
          }}
          // Set label
          memset(&data[{expected_onehot[0]._id}], 0, {NUM_DIGITS}*sizeof data[0]);
          data[{expected_onehot[0]._id}+label] = 1.0L;
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
          Py_RETURN_NONE;
    }}

    PyObject* backward_wrapper(PyObject* module) {{
          backward();
          Py_RETURN_NONE;
    }}

    PyObject* update_wrapper(PyObject* module, PyObject* step_obj) {{
          int step = PyLong_AsLong(step_obj);
          if (step < 0 && PyErr_Occurred()) {{
                return NULL;
          }}
          update(step);
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
          if (i >= {DIM}) {{
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
          if (i >= {DIM}) {{
                  PyErr_Format(PyExc_TypeError, "index out of bounds");
                  return NULL;
          }}
          return PyFloat_FromDouble(grad[i]);
    }}

    static PyMethodDef nn_methods[] = {{
          {{ "forward", (PyCFunction)forward_wrapper, METH_FASTCALL, "doc" }},
          {{ "zero_grad", (PyCFunction)zero_grad_wrapper, METH_NOARGS, "doc" }},
          {{ "backward", (PyCFunction)backward_wrapper, METH_NOARGS, "doc" }},
          {{ "update", update_wrapper, METH_O, "doc" }},
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

losses = collections.deque((), maxlen=16)


def loss_changing():
    eps = 1
    if len(losses) < losses.maxlen:
        # We are early in the process; keep going.
        return True
    if any(math.isnan(loss) or math.isinf(loss) for loss in losses):
        # Stop iteration; something went wrong.
        return False
    return max(losses) - min(losses) < eps


parser = argparse.ArgumentParser()
parser.add_argument("--compiled", action="store_true")
args = parser.parse_args()

if args.compiled:
    source_file = timer(lambda: write_code(), "Writing C code...")
    # TODO(max): Bring back Extension stuff and customize compiler using
    # https://shwina.github.io/custom-compiler-linker-extensions/
    lib_file = "nn.so"
    include_dir = sysconfig.get_python_inc()
    timer(
        lambda: os.system(f"tcc -shared -fPIC -I{include_dir} nn.c -o {lib_file}"),
        "Compiling extension...",
    )
    spec = importlib.machinery.ModuleSpec("nn", None, origin=lib_file)
    nn = timer(lambda: _imp.create_dynamic(spec), "Loading extension...")
    print("Training...")
    nrounds = 0
    while loss_changing():
        im = next(db)
        loss = nn.forward(im.label, im.pixels[:DIM])
        print(f"...round {nrounds:4d} loss {loss:.2f}")
        losses.append(loss)
        nn.zero_grad()
        nn.backward()
        nn.update(nrounds)
        nrounds += 1
else:
    nn = model
    print("Training...")
    nrounds = 0
    while loss_changing():
        im = next(db)
        out = nn(im.pixels[:DIM])
        expected_onehot = [Value(0, (), "input") for _ in range(NUM_DIGITS)]
        expected_onehot[im.label] = 1.0
        loss = sum((exp - act) ** 2 for exp, act in zip(expected_onehot, out))
        print(f"...round {nrounds:4d} loss {loss.data:.2f}")
        losses.append(loss.data)
        nn.zero_grad()
        loss.backward()
        # nn.update(nrounds)
        learning_rate = 1.0 - 0.9*nrounds/100
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        nrounds += 1
