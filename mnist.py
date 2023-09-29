import random
import struct
import sys
import time
import math

from rpython.rlib import jit
from rpython.rlib import rrandom


random = rrandom.Random()


class Value(object):
    """ stores a single scalar value and its gradient """
    _attrs_ = ['data', 'grad', '_visited']
    _op = ''

    def __init__(self, data):
        self.data = data
        self.grad = 0
        self._visited = False

    def _backward(self): pass

    def _forward(self): pass

    def add(self, other):
        out = AddValue(self.data + other.data, self, other)
        return out

    def mul(self, other):
        out = MulValue(self.data * other.data, self, other)
        return out

    def sub(self, other):
        return self.add(other.mul(Value(-1)))

    def pow(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = PowValue(math.pow(self.data, other), self)
        out.exponent = other
        return out

    def div(self, other): # self / other
        return self.mul(other.pow(-1))

    def relu(self):
        out = ReluValue((self.data > 0) * self.data, self)
        return out

    def log(self):
        out = LogValue(math.log(self.data), self)
        return out

    def exp(self):
        out = ExpValue(math.exp(self.data), self)
        return out

    def topo(self):
        # topological order all of the children in the graph
        topo = []
        self._build_topo(topo)
        return topo

    @jit.unroll_safe
    def _build_topo(self, topo):
        if not self._visited:
            self._visited = True
            topo.append(self)
            self._build_topo_recurse(topo)

    def _build_topo_recurse(self, topo):
        pass
    #    for child in self._prev:
    #            child._build_topo(topo)

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

class UnaryValue(Value):
    def __init__(self, data, prev0):
        Value.__init__(self, data)
        self._prev0 = prev0

    def _build_topo_recurse(self, topo):
        self._prev0._build_topo(topo)


class BinaryValue(Value):
    def __init__(self, data, prev0, prev1):
        Value.__init__(self, data)
        self._prev0 = prev0
        self._prev1 = prev1

    def _build_topo_recurse(self, topo):
        self._prev0._build_topo(topo)
        self._prev1._build_topo(topo)


class AddValue(BinaryValue):
    _op = '+'
    def _forward(self):
        self.data = self._prev0.data + self._prev1.data
    def _backward(out):
        self, other = out._prev0, out._prev1
        self.grad += out.grad
        other.grad += out.grad

class MulValue(BinaryValue):
    _op = '*'
    def _forward(self):
        self.data = self._prev0.data * self._prev1.data
    def _backward(out):
        self, other = out._prev0, out._prev1
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

class PowValue(UnaryValue):
    _op = 'pow'
    def _forward(self):
        self.data = math.pow(self._prev0.data, self.exponent)
    def _backward(out):
        self = out._prev0
        self.grad += (out.exponent * math.pow(self.data, out.exponent - 1)) * out.grad

class ReluValue(UnaryValue):
    _op = 'relu'
    def _forward(self):
        self.data = (self._prev0.data > 0) * self._prev0.data
    def _backward(out):
        self = out._prev0
        self.grad += (out.data > 0) * out.grad


class LogValue(UnaryValue):
    _op = 'log'
    def _forward(self):
        self.data = math.log(self._prev0.data)
    def _backward(out):
        self = out._prev0
        self.grad += 1/self.data * out.grad

class ExpValue(UnaryValue):
    _op = 'exp'
    def _forward(self):
        self.data = math.exp(self._prev0.data)
    def _backward(out):
        self = out._prev0
        self.grad += math.exp(self.data) * out.grad

class Max(BinaryValue):
    _op = 'max'
    def __init__(self, left, right):
        left_bigger = float(left.data > right.data)
        # bad branch-free way to write max :-(
        result = left_bigger * left.data + (1.0 - left_bigger) * right.data
        BinaryValue.__init__(self, result, left, right)

    def _forward(self):
        # bad branch-free way to write max :-(
        left, right = self._prev0, self._prev1
        left_bigger = float(left.data > right.data)
        self.data = left_bigger * left.data + (1.0 - left_bigger) * right.data

    def _backward(self):
        left, right = self._prev0, self._prev1
        left_bigger = float(left.data > right.data)
        left.grad += left_bigger * self.grad
        right.grad += (1.0 - left_bigger) * self.grad

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
        self.w = [Value(random.random()) for _ in range(nin)] # op=weight
        self.b = Value(0) # op=bias
        self.nonlin = nonlin
        self._parameters = (self.w + [self.b])[:]

    @jit.unroll_safe
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    @jit.unroll_safe
    def evalneuron(self, x):
        # assert len(self.w) == len(x), f"input of size {len(x)} with {len(self.w)} weights"
        result = self.b
        for i in range(len(x)):
            result = result.add(self.w[i].mul(x[i]))
        # for wi, xi in zip(self.w, x):
        #     result = result.add(wi.mul(xi))
        return result.relu() if self.nonlin else result
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
    _immutable_fields_ = ['labels', 'pixels']
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
    allres = []
    currlist = []
    for x in iterable:
        currlist.append(x)
        if len(currlist) == n:
            allres.append(currlist)
            currlist = []
    if len(currlist) < n:
        for i in range(len(currlist), n):
            currlist.append(fillvalue)
        allres.append(currlist)
    return allres

def stable_softmax(output):
    max_ = output[0]
    for i in range(1, len(output)):
        max_ = Max(max_, output[i])
    shiftx = [o.sub(max_) for o in output]
    exps = [o.exp() for o in shiftx]
    sum_ = exps[0]
    for index in range(1, len(exps)):
        sum_ = sum_.add(exps[index])
    return [o.div(sum_) for o in exps]


NUM_DIGITS = 10

def randbelow(n):
    return int(n * random.random())

def shuffle(x):
    for i in range(len(x)-1, 0, -1):
        # pick an element in x[:i+1] with which to exchange x[i]
        j = randbelow(i + 1)
        x[i], x[j] = x[j], x[i]

@jit.unroll_safe
def loss_of(model, inp_, expected_onehot):
    output = model.evalmlp(inp_)
    softmax_output = stable_softmax(output)
    result = Value(0.0)
    for i in range(len(expected_onehot)):
        exp = expected_onehot[i]
        act = softmax_output[i]
        result = result.add(act.add(Value(0.0001)).log().mul(exp))
    return result.mul(Value(-1))

def make_main():
    inp_ = [Value(0.) for _ in range(DIM)]
    model = timer(lambda: MLP(DIM, [50, NUM_DIGITS]), "Building model...")
    expected_onehot = [Value(0.) for _ in range(NUM_DIGITS)]
    loss = loss_of(model, inp_, expected_onehot)
    topo = loss.topo()
    params = model.parameters()
    non_params = [p for p in topo if p not in params]
    reverse_topo = loss.topo()[::-1]
    db = list(images("train-images-idx3-ubyte", "train-labels-idx1-ubyte"))
    driver = jit.JitDriver(greens=[], reds='auto')

    @jit.dont_look_inside
    def shuffle_and_group(batch_size, num_training_images):
        if num_training_images < 0:
            l = db[:]
        else:
            l = db[:num_training_images]
        shuffle(l)
        return grouper(batch_size, l)

    def forward(image):
        for e in expected_onehot:
            e.data = 0.
        expected_onehot[image.label].data = 1.
        for inp, pix in zip(inp_, image.pixels):
            # TODO(max): Should divide by 255
            inp.data = ord(pix)
        for node in topo:
            node._forward()
        return loss.data

    def backward():
        for node in non_params:
            node.grad = 0.
        loss.grad = 1.
        for node in reverse_topo:
            node._backward()

    def main(args):
        print("Training...")
        for i in range(len(args)):
            if args[i] == "--jit":
                if len(args) == i + 1:
                    print "missing argument after --jit"
                    return 2
                jitarg = args[i + 1]
                del args[i:i+2]
                jit.set_user_param(None, jitarg)
                break
        if len(args) >= 2:
            num_epochs = int(args[1])
            del args[1]
        else:
            num_epochs = 100
        if len(args) >= 2:
            batch_size = int(args[1])
            del args[1]
        else:
            batch_size = 10
        if len(args) >= 2:
            num_training_images = int(args[1])
            del args[1]
        else:
            num_training_images = -1
        for epoch in range(num_epochs):
            print epoch
            epoch_loss = 0.
            before = time.time()
            batches = shuffle_and_group(batch_size, num_training_images)
            for batch_idx, batch in enumerate(batches):
                driver.jit_merge_point()
                print "   ", batch_idx, "of", len(batches)
                # zero grad
                for node in topo:
                    node.grad = 0.
                    node._visited = False
                jit.promote(len(batch))
                for im in batch:
                    if im is not None:
                        epoch_loss += forward(im)
                        backward()
                for p in model.parameters():
                    p.data -= 0.1 * p.grad
            after = time.time()
            delta = after - before
            epoch_loss /= len(db)
            print "...epoch %s loss %s took %s sec)" % (epoch, epoch_loss, delta)
        return 0
    return main

def target(*args):
    return make_main()

if __name__ == '__main__':
    try:
        make_main()(sys.argv)
    except:
        import pdb;pdb.xpm()
