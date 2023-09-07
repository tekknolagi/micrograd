import struct
import time
import itertools
import random
import math
import numpy as np
from micrograd.engine import Value


random.seed(1337)


class Tensor:
    def __init__(self, data, children=(), op=''):
        assert hasattr(data, 'shape')
        assert len(data.shape) > 1
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

    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward
        return out

    def __add__(self, other):
        if isinstance(other, float):
            other = Tensor(np.full(self.data.shape, other), (), '')
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        out = Tensor(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad += out.grad
            other.grad += -out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += np.exp(self.data) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')
        def _backward():
            self.grad += 1/self.data * out.grad
        out._backward = _backward
        return out

    def sum(self):
        x = np.sum(self.data)
        out = Value(np.sum(self.data), (self,), 'sum')
        def _backward():
            self.grad += out.grad
        out._backward = _backward
        return out

    def max(self):
        out = Tensor(np.max(self.data), (self,), 'max')
        def _backward():
            # TODO(max): Check backprop
            self.grad += out.grad
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
        pixels = list(pixels)
        pixels = np.fromiter(pixels, np.float64, len(pixels))
        self.pixels = np.expand_dims(pixels, axis=0)


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


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


def stable_softmax(output):
    exps = (output - output.max()).exp()
    return exps/exps.sum()


if __name__ == "__main__":
    print("Loading images...")
    db = list(images("train-images-idx3-ubyte", "train-labels-idx1-ubyte"))
    print("Building model...")
    model = MLP(784, [512, 10])
    print("Training...")
    num_epochs = 100
    batch_size = 200
    for epoch in range(num_epochs):
        epoch_loss = 0.
        before = time.perf_counter()
        shuffled = db.copy()
        random.shuffle(shuffled)
        for batch_idx, batch in enumerate(grouper(batch_size, shuffled)):
            model.zero_grad()
            batch_loss = 0.
            num_correct = 0.
            for im in batch:
                logits = model(Tensor(im.pixels, (), 'input'))
                probs = stable_softmax(logits)
                exp = np.identity(10)[im.label]
                exp = Tensor(np.expand_dims(exp, axis=0), (), 'input')
                loss = (exp*(probs+0.0001).log()).sum()
                batch_loss += loss.data
                epoch_loss += loss.data
                loss.backward()
            batch_loss /= batch_size
            accuracy = num_correct/batch_size
            for p in model.parameters():
                p.data -= 0.1 * p.grad
            if batch_idx % 20 == 0:
                print(f"batch {batch_idx:4d} loss {batch_loss:.2f} acc {accuracy:.2f}")
        after = time.perf_counter()
        delta = after - before
        epoch_loss /= len(db)
        print(f"...epoch {epoch:4d} loss {epoch_loss:.2f} ({len(db)/delta} images/sec)")
