import struct
import random
import numpy as np


random.seed(1337)


class Tensor:
    def __init__(self, data, children, op):
        assert isinstance(data, np.ndarray)
        self.data = data
        # TODO(max): Is grad the same dims as data?
        self.grad = np.zeros_like(data)
        self._prev = children
        self._op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(self.data * (self.data > 0), (self,), 'ReLU')
        def _backward():
            self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Layer(Module):

    def __init__(self, nin, nout, nonlin):
        self.W = Tensor(np.random.uniform(-1,1,(nin,nout)), (), 'W')
        b = np.random.uniform(-1,1,(nout,))
        self.b = Tensor(np.expand_dims(b, axis=0), (), 'b')
        self.nonlin = nonlin

    def __call__(self, x):
        act = (x @ self.W) + self.b
        return act.relu() if self.nonlin else act

    def parameters(self):
        return [self.W, self.b]

    def __repr__(self):
        return f"Layer of {self.W.data.shape}"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"



IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
PIXEL_LENGTH = IMAGE_HEIGHT * IMAGE_WIDTH
DIM = PIXEL_LENGTH



class image:
    def __init__(self, label, pixels):
        self.label = label
        self.pixels = np.expand_dims(np.array([float(p) for p in pixels]), axis=0)


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



print("Loading images...")
db = images("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
print("Building model...")
model = MLP(784, [512, 10])
print("Making tensor...")
im = Tensor(next(db).pixels, (), 'input')
print(model)
print("Evaluating model...")
out = model(im)
print(out)
print("Backprop...")
out.backward()
print(im)
