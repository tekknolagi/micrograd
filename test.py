import _imp
import argparse
import functools
import importlib
import itertools
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
from micrograd.engine import Value, Max


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
        label_bytes = self.labels.read(1)
        assert label_bytes
        label = int.from_bytes(label_bytes, "big")
        pixels = self.images.read(PIXEL_LENGTH)
        assert pixels
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


def stable_softmax(output):
    max_ = functools.reduce(Max, output)
    shiftx = [o-max_ for o in output]
    exps = [o.exp() for o in shiftx]
    sum_ = sum(exps)
    return [o/sum_ for o in exps]


def optimize_ir(output):
    topo = output.topo()
    num_fma = 0
    for o in topo:
        # Find a*b+c
        if o._op != '+':
            continue
        left, right = o._prev
        if left._op == '*':
            a, b = left._prev
            num_fma += 1
        elif right._op == '*':
            a, b = right._prev
            num_fma += 1
        else:
            continue
    # TODO(max): Figure out how to replace nodes in-place. Maybe need to build
    # a mapping of node->replacement and then visit all uses.
    print(f"...found {num_fma} FMA opportunities")
    return output


NUM_DIGITS = 10
model = timer(lambda: nn_interp.MLP(DIM, [50, NUM_DIGITS]), "Building model...")
# NOTE: It's important that input are all in sequence right next to one another
# so we can set the input in training
inp = [Value(0, (), "input") for _ in range(DIM)]
assert [i._id for i in inp] == list(range(inp[0]._id, inp[0]._id + len(inp)))
output = model(inp)
softmax_output = stable_softmax(output)
# NOTE: It's important that expected_onehot are all in sequence right next to
# one another so we can set the label in training
expected_onehot = [Value(0, (), "input") for _ in range(NUM_DIGITS)]
assert [exp._id for exp in expected_onehot] == list(
    range(expected_onehot[0]._id, expected_onehot[0]._id + len(expected_onehot))
)
loss = -sum(exp*(act+0.0001).log() for exp, act in zip(expected_onehot, softmax_output))
loss = optimize_ir(loss)
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
    double data[{num_nodes}];
    double grad[{num_nodes}];
    static inline __attribute__((always_inline)) double relu(double x) {{
        return fmax(x, 0);
    }}
    static inline __attribute__((always_inline)) double clip(double x) {{
        return x;
    }}
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
            data[{inp[0]._id}+i] = ((double)(unsigned char)buf[i])/255;
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
            params = frozenset(model.parameters())
            for o in topo:
                if o not in params:
                    print(f"grad[{o._id}] = 0;", file=f)
            print(f"{loss.getgrad()} = 1;", file=f)
            for o in reversed(topo):
                for line in o.backward_compile():
                    print(line, file=f)
            print("}", file=f)
            print("void update(int step, int batch_size) {", file=f)
            print("double learning_rate = 0.1;", file=f)
            print("int idx = 0;", file=f)
            for o in model.parameters():
                assert o._op in ('weight', 'bias'), repr(o._op)
                assert '[' in o.getgrad()
                print(f"{{ double grad_update = learning_rate * {o.getgrad()} / ((double)batch_size);", file=f)
                print("assert(!isnan(grad_update));", file=f)
                print("assert(!isinf(grad_update));", file=f)
                print(f"data[{o._id}] -= grad_update; idx++; }}", file=f)
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
          int batch_size = PyLong_AsLong(batch_size_obj);
          if (batch_size < 0 && PyErr_Occurred()) {{
                return NULL;
          }}
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


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


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
        lambda: os.system(f"tcc -DNDEBUG -g -shared -fPIC -I{include_dir} nn.c -o {lib_file}"),
        "Compiling extension...",
    )
spec = importlib.machinery.ModuleSpec("nn", None, origin=lib_file)
nn = timer(lambda: _imp.create_dynamic(spec), "Loading extension...")
print("Training...")
num_epochs = 100
traindb = list(images("train-images-idx3-ubyte", "train-labels-idx1-ubyte"))
testdb = list(images("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"))
def argmax(output):
    return max(enumerate(output), key=lambda x: x[1])[0]
def accuracy():
    num_correct = 0
    for im in testdb:
        nn.forward(im.label, im.pixels)
        guess = argmax([nn.data(o._id) for o in output])
        if guess == im.label:
            num_correct += 1
    return num_correct/len(testdb)
batch_size = 20
for epoch in range(num_epochs):
    epoch_loss = 0
    before = time.perf_counter()
    shuffled = traindb.copy()
    random.shuffle(shuffled)
    for batch_idx, batch in enumerate(grouper(batch_size, shuffled)):
        nn.zero_grad()
        batch_loss = 0
        for im in batch:
            im_loss = nn.forward(im.label, im.pixels)
            outs = [nn.data(o._id) for o in softmax_output]
            assert not any(math.isnan(o) for o in outs)
            assert not math.isnan(im_loss)
            assert not math.isinf(im_loss)
            batch_loss += im_loss
            epoch_loss += im_loss
            nn.backward()
        batch_loss /= batch_size
        nn.update(epoch, batch_size)
        if batch_idx % 20 == 0:
            print(f"batch {batch_idx:4d} loss {batch_loss:.2f}")
    after = time.perf_counter()
    delta = after - before
    epoch_loss /= len(traindb)
    print(f"...epoch {epoch:4d} loss {epoch_loss:.2f} acc {accuracy():.2f} (took {delta} sec)")
