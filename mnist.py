import functools
import itertools
import random
import struct
import sys
import time
import math

from rpython.rlib import jit
from rpython.rlib import rrandom


random = rrandom.Random()


@jit.unroll_safe
def build_topo(topo, v):
    if not v._visited:
        v._visited = True
        for child in v._prev:
            build_topo(topo, child)
        topo.append(v)

class Value(object):
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=[], _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._prev = _children
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self._visited = False

    def _backward(self): pass

    def add(self, other):
        out = AddValue(self.data + other.data, [self, other], '+')
        return out

    def mul(self, other):
        out = MulValue(self.data * other.data, [self, other], '*')
        return out

    def sub(self, other):
        return self.add(other.mul(Value(-1)))

    def pow(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        assert other == 2
        out = PowValue(self.data * self.data, [self], '**'+str(other))
        return out

    def div(self, other): # self / other
        return self.mul(other.pow(-1))

    def relu(self):
        out = ReluValue(0 if self.data < 0 else self.data, [self], 'ReLU')
        return out

    def log(self):
        out = LogValue(math.log(self.data), [self], 'log')
        return out

    def exp(self):
        out = ExpValue(math.exp(self.data), [self], 'exp')
        return out

    def topo(self):
        # topological order all of the children in the graph
        topo = []
        build_topo(topo, self)
        return topo

    @jit.unroll_safe
    def backward(self):
        topo = self.topo()
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

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
        return "Value(data=%s, grad=%s)" % (self.data, self.grad)


class AddValue(Value):
    def _backward(out):
        self, other = out._prev
        self.grad += out.grad
        other.grad += out.grad

class MulValue(Value):
    def _backward(out):
        self, other = out._prev
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

class PowValue(Value):
    def _backward(out):
        self, = out._prev
        other = int(out._op[2:])
        assert other == 2
        #self.grad += (other * self.data**(other-1)) * out.grad
        self.grad += (other * self.data) * out.grad

class ReluValue(Value):
    def _backward(out):
        self, = out._prev
        self.grad += (out.data > 0) * out.grad


class LogValue(Value):
    def _backward(out):
        self, = out._prev
        self.grad += 1/self.data * out.grad

class ExpValue(Value):
    def _backward(out):
        self, = out._prev
        self.grad += math.exp(self.data) * out.grad

class Max(Value):
    def __init__(self, left, right):
        Value.__init__(self, max(left.data, right.data), [left, right], 'max')
    # TODO(max): Copy _backward from code/micrograd


class Module(object):

    @jit.unroll_safe
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            p._visited = False

    @jit.elidable
    def parameters(self):
        return []

class Neuron(object):
    _immutable_fields_ = ['w[*]', '_parameters[*]', 'b', 'nonlin']

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.random(), [], 'weight') for _ in range(nin)]
        self.b = Value(0, [], 'bias')
        self.nonlin = nonlin
        self._parameters = (self.w + [self.b])[:]

    @jit.unroll_safe
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    @jit.unroll_safe
    def evalneuron(self, x):
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
        return self._parameters

    def __repr__(self):
        return "%sNeuron(%s)" % ('ReLU' if self.nonlin else 'Linear', len(self.w))

class Layer(Module):
    _immutable_fields_ = ['neurons[*]', '_parameters[*]']

    def __init__(self, nin, nout, nonlin):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]
        self._parameters = [p for n in self.neurons for p in n.parameters()][:]

    @jit.unroll_safe
    def evallayer(self, x):
        out = [n.evalneuron(x) for n in self.neurons]
        return out
        # return out[0] if len(out) == 1 else out

    def parameters(self):
        return self._parameters

    def __repr__(self):
        return "Layer of [%s]" % (', '.join(str(n) for n in self.neurons))

class MLP(Module):
    _immutable_fields_ = ['layers[*]', '_parameters[*]']

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
        self._parameters = [p for layer in self.layers for p in layer.parameters()][:]

    @jit.unroll_safe
    def evalmlp(self, x):
        for layer in self.layers:
            x = layer.evallayer(x)
        return x

    def parameters(self):
        return self._parameters

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


def loss_of(model, image):
    output = model.evalmlp(image.pixels)
    softmax_output = stable_softmax(output)
    expected_onehot = [0. for _ in range(NUM_DIGITS)]
    expected_onehot[image.label] = 1.
    result = Value(0.0)
    for exp, act in zip(expected_onehot, softmax_output):
        result = result.add(act.add(0.0001).log().mul(exp))
    return result.mul(Value(-1))


def main():
    model = timer(lambda: MLP(DIM, [50, NUM_DIGITS]), "Building model...")
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
                p._visited = False
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
