import _imp
import importlib
import math
import micrograd
import os
import random
import shutil
import struct
import tempfile
from distutils import sysconfig
from micrograd import nn as nn_interp
from micrograd.engine import Value


random.seed(1337)


class image:
    def __init__(self, label, pixels):
        self.label = label
        self.pixels = list(pixels)


IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
PIXEL_LENGTH = IMAGE_HEIGHT * IMAGE_WIDTH


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


print("Opening images...")
db = images("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
print("Building model...")
# dim = 784
# model = nn_interp.MLP(dim, [512, 10])
dim = 784
model = nn_interp.MLP(dim, [10, 10])
inp = [Value(0, (), 'input') for _ in range(dim)]
out = model(inp)
expected_onehot = [Value(0, (), 'input') for _ in range(10)]
expected_onehot[3] = Value(1, (), 'input')
loss = sum((exp-act)**2 for exp,act in zip(expected_onehot, out))
topo = loss.topo()

print("Writing C code...")
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
double data[{micrograd.engine.counter}];
double grad[{micrograd.engine.counter}];
double relu(double x) {{ if (x < 0) {{ return 0; }} else {{ return x; }} }}
void init() {{
        """,
            file=f,
        )
        for o in model.parameters():
            print(f"data[{o._id}] = {o.data}L;", file=f)
        print("}", file=f)
        print(f"void set_input(PyObject* input_data) {{", file=f)
        print("PyObject* item_obj; double item_double;", file=f)
        for idx, o in enumerate(inp):
            # TODO(max): Read image and also update label in loss
            print(f"""\
            item_obj = PyList_GetItem(input_data, {idx});
            if (item_obj == NULL) {{
                abort();
            }}
            item_double = PyFloat_AsDouble(item_obj);
            if (item_double < 0 && PyErr_Occurred()) {{
                abort();
            }}
            data[{o._id}] = item_double;
            """, file=f)
        print("}", file=f)
        print("void forward() {", file=f)
        for o in topo:
            lines = o.compile()
            if lines:
                print("\n".join(lines), file=f)
        print("}", file=f)
        # Don't just zero the parameters; Karpathy can get away with that
        # because he rebuilds the entire graph every time, but we don't.
        print("void zero_grad() { memset(grad, 0, sizeof grad); }", file=f)
        print("void backward() {", file=f)
        print(f"grad[{loss._id}] = 1;", file=f)
        for o in reversed(topo):
            lines = o.backward_compile()
            if lines:
                print("\n".join(lines), file=f)
        print("}", file=f)
        print("void update(int step) {", file=f)
        # TODO(max): It's not always 100; is this hard-coded for number of
        # training rounds in Karpathy's code?
        print("double learning_rate = 1.0L - (0.9L * (double)step) / 100.0L;", file=f)
        for o in model.parameters():
            print(f"data[{o._id}] -= learning_rate * grad[{o._id}];", file=f)
        print("}", file=f)
        # print("\n".join(n.compile()), file=f)
        print(
            f"""
#include <Python.h>

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
      if (!PyList_CheckExact(pixels_obj)) {{
            PyErr_Format(PyExc_TypeError, "expected list");
            return NULL;
      }}
      if (PyList_Size(pixels_obj) != {dim}) {{
            PyErr_Format(PyExc_TypeError, "expected list of size {dim}");
            return NULL;
      }}
      int label = PyLong_AsLong(label_obj);
      if (label < 0 && PyErr_Occurred()) {{
            return NULL;
      }}
      // TODO(max): Set label
      set_input(pixels_obj);
      forward();
      // TODO(max): Make this able to return multiple outputs?
      double loss = data[{loss._id}];
      return PyFloat_FromDouble(loss);
}}

PyObject* zero_grad_wrapper(PyObject* module) {{
      zero_grad();
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

static PyMethodDef nn_methods[] = {{
      {{ "forward", (PyCFunction)forward_wrapper, METH_FASTCALL, "doc" }},
      {{ "zero_grad", (PyCFunction)zero_grad_wrapper, METH_NOARGS, "doc" }},
      {{ "backward", (PyCFunction)backward_wrapper, METH_NOARGS, "doc" }},
      {{ "update", update_wrapper, METH_O, "doc" }},
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

    print("Compiling extension...")
    include_dir = sysconfig.get_python_inc()
    lib_file = f"{build_dir}/nn.so"
    os.system(f"tcc -shared -fPIC -I{include_dir} {file_path} -o {lib_file}")
    shutil.copyfile(f.name, "nn.c")
    shutil.copyfile(lib_file, "nn.so")

    print("Loading extension...")
    spec = importlib.machinery.ModuleSpec("nn", None, origin=lib_file)
    nn = _imp.create_dynamic(spec)


n = 0
nrounds = 0
NUM_LOSSES = 16
EPS = 1
losses = [1] * NUM_LOSSES
losses[1] = 2
loss_idx = 0


def add_loss(loss):
    global loss_idx
    losses[loss_idx] = loss
    loss_idx = (loss_idx + 1) % NUM_LOSSES


def loss_changing():
    count = 0
    for loss in losses:
        if math.isnan(loss) or math.isinf(loss):
            return False
        if abs(loss-losses[loss_idx]) < EPS:
            count += 1
    return count < NUM_LOSSES


nrounds = 0
while loss_changing():
    im = next(db)
    loss = nn.forward(im.label, im.pixels)
    print(f"round {nrounds:4d} loss {loss:.2f}")
    add_loss(loss)
    nn.zero_grad()
    nn.backward()
    nn.update(nrounds)
    nrounds += 1
