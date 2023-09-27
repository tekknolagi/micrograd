import functools
import itertools
import random
import struct
import sys
import time
import math

from rpython.rlib import rrandom


random = rrandom.Random()


def _backward_add(out):
    self, other = out._prev
    self.grad += out.grad
    other.grad += out.grad


def _backward_mul(out):
    self, other = out._prev
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad


def _backward_pow(out):
    self, = out._prev
    other = int(out._op[2:])
    self.grad += (other * self.data**(other-1)) * out.grad


def _backward_relu(out):
    self, = out._prev
    self.grad += (out.data > 0) * out.grad


def _backward_log(out):
    self, = out._prev
    self.grad += 1/self.data * out.grad


def _backward_exp(out):
    self, = out._prev
    self.grad += math.exp(self.data) * out.grad


def build_topo(visited, topo, v):
    if v not in visited:
        visited[v] = None
        for child in v._prev:
            build_topo(visited, topo, child)
        topo.append(v)

class Value(object):
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=[], _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda v: None
        self._prev = _children
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def add(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, [self, other], '+')
        out._backward = _backward_add
        return out

    def mul(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, [self, other], '*')
        out._backward = _backward_mul
        return out

    def sub(self, other):
        return self.add(other.mul(-1))

    def pow(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, [self], '**'+str(other))
        out._backward = _backward_pow
        return out

    def div(self, other): # self / other
        return self.mul(other.pow(-1))

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, [self], 'ReLU')
        out._backward = _backward_relu
        return out

    def log(self):
        out = Value(math.log(self.data), [self], 'log')
        out._backward = _backward_log
        return out

    def exp(self):
        out = Value(math.exp(self.data), [self], 'exp')
        out._backward = _backward_exp
        return out

    def topo(self):
        # topological order all of the children in the graph
        topo = []
        visited = {}
        build_topo(visited, topo, self)
        return topo


    def backward(self):
        topo = self.topo()
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward(v)

    # def __neg__(self): # -self
    #     return self * -1

    # def __radd__(self, other): # other + self
    #     return self + other

    # def __sub__(self, other): # self - other
    #     return self + (-other)

    # def __rsub__(self, other): # other - self
    #     return other + (-self)

    # def __rmul__(self, other): # other * self
    #     return self * other

    # def __rtruediv__(self, other): # other / self
    #     return other * self**-1

    def __repr__(self):
        return "Value(data=%, grad=%)" % (self.data, self.grad)


class Max(Value):
    def __init__(self, left, right):
        Value.__init__(self, max(left.data, right.data), [left, right], 'max')
    # TODO(max): Copy _backward from code/micrograd


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.random(), [], 'weight') for _ in range(nin)]
        self.b = Value(0, [], 'bias')
        self.nonlin = nonlin

    def eval(self, x):
        # assert len(self.w) == len(x), f"input of size {len(x)} with {len(self.w)} weights"
        result = Value(0.0)
        for i in range(len(x)):
            result = result.add(self.w[i].mul(x[i]))
        # for wi, xi in zip(self.w, x):
        #     result = result.add(wi.mul(xi))
        act = result.add(self.b)
        return act.relu() if self.nonlin else act
        # act = sum([wi*xi for wi,xi in zip(self.w, x)], self.b)
        # return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return "%sNeuron(%s)" % ('ReLU' if self.nonlin else 'Linear', len(self.w))

class Layer(Module):

    def __init__(self, nin, nout, nonlin):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def eval(self, x):
        out = [n.eval(x) for n in self.neurons]
        return out
        # return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return "Layer of [%s]" % (', '.join(str(n) for n in self.neurons))

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def eval(self, x):
        for layer in self.layers:
            x = layer.eval(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return "MLP of [%s]" % (', '.join(str(layer) for layer in self.layers))


sys.setrecursionlimit(20000)


class image(object):
    def __init__(self, label, pixels):
        self.label = label
        self.pixels = [ord(x) for x in pixels]


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
        label = ord(label_bytes)
        pixels = self.images.read(PIXEL_LENGTH)
        assert pixels
        self.idx += 1
        return image(label, pixels)

    def __iter__(self):
        return self

    def next(self):
        if self.idx >= self.num_images:
            raise StopIteration
        return self.read_image()

    def num_left(self):
        return self.num_images - self.idx


def timer(lam, msg=""):
    print msg,
    before = time.time()
    result = lam()
    after = time.time()
    delta = after - before
    print "({delta:.2f} s)".format(delta=delta)
    return result



def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    it = iter(iterable)
    while 1:
        res = []
        for i in range(n):
            res.append(next(it, fillvalue))
        yield tuple(res)
        if res[-1] == fillvalue:
            break


def stable_softmax(output):
    max_ = functools.reduce(Max, output)
    shiftx = [o.sub(max_) for o in output]
    exps = [o.exp() for o in shiftx]
    sum_ = exps[0]
    for x in exps[1:]:
        sum_ = sum_.add(x)
    return [o.div(sum_) for o in exps]


NUM_DIGITS = 10
model = timer(lambda: MLP(DIM, [50, NUM_DIGITS]), "Building model...")


def loss_of(model, image):
    output = model.eval(image.pixels)
    softmax_output = stable_softmax(output)
    expected_onehot = [0. for _ in range(NUM_DIGITS)]
    expected_onehot[image.label] = 1.
    result = Value(0.0)
    for exp, act in zip(expected_onehot, softmax_output):
        result = result.add(act.add(0.0001).log().mul(exp))
    return result.mul(-1)


def main():
    print("Training...")
    num_epochs = 100
    db = list(images("train-images-idx3-ubyte", "train-labels-idx1-ubyte"))
    batch_size = 10
    for epoch in range(num_epochs):
        print epoch
        epoch_loss = 0.
        before = time.time()
        shuffled = db[:]
        #random.shuffle(shuffled)
        for batch_idx, batch in enumerate(grouper(batch_size, shuffled)):
            print "   ", batch_idx
            for p in model.parameters():
                p.grad = 0.0
            loss = Value(0.0)
            for im in batch:
                loss = loss.add(loss_of(model, im))
            loss.backward()
            epoch_loss += loss.data
            for p in model.parameters():
                p.data -= 0.1 * p.grad
        after = time.time()
        delta = after - before
        epoch_loss /= len(db)
        print "...epoch {epoch:4d} loss {epoch_loss:.2f} (took {delta} sec)".format(epoch=epoch, epoch_loss=epoch_loss, delta=delta)

if __name__ == '__main__':
    try:
        main()
    except:
        import pdb;pdb.xpm()
