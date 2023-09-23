import functools
import itertools
import random
import struct
import sys
import time
import math
from micrograd import nn as nn_interp
from micrograd.engine import Value, Max


random.seed(1337)
sys.setrecursionlimit(20000)


class image:
    def __init__(self, label, pixels):
        self.label = label
        self.pixels = [float(i)/255 for i in pixels]


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


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


def stable_softmax(x):
    max_ = functools.reduce(Max, x)
    shiftx = [o-max_ for o in x]
    numerator = [o.exp() for o in shiftx]
    denominator = sum(numerator)
    return [o/denominator for o in numerator]


NUM_DIGITS = 10
model = timer(lambda: nn_interp.MLP(DIM, [50, NUM_DIGITS]), "Building model...")


def logsumexp(x):
    return (sum(o.exp() for o in x)+1e-15).log()


def mean(x):
    return sum(x)/len(x)


def loss_of(model, expected_onehot, image):
    output = model(image.pixels)
    softmax_output = stable_softmax(output)
    result = -sum(exp*(act+0.0001).log() for exp, act in zip(expected_onehot, softmax_output))
    return result


db = list(images("train-images-idx3-ubyte", "train-labels-idx1-ubyte"))
im = image(db[0].label, db[0].pixels)
inp_ = [Value(pixel) for pixel in im.pixels]
im.pixels = inp_
expected_onehot = [Value(0.) for _ in range(NUM_DIGITS)]
loss = loss_of(model, expected_onehot, im)
topo = loss.topo()
reverse_topo = reversed(topo)
params = model.parameters()
non_params = set(topo)-set(params)


def set_input(label, pixels):
    for exp in expected_onehot:
        exp.data = 0.0
    expected_onehot[label].data = 1.0
    for inp, pix in zip(inp_, pixels):
        inp.data = pix


def run():
    print("Training...")
    num_epochs = 100
    batch_size = 10
    for epoch in range(num_epochs):
        epoch_loss = 0.
        before = time.perf_counter()
        shuffled = db.copy()
        random.shuffle(shuffled)
        for batch_idx, batch in enumerate(grouper(batch_size, shuffled)):
            batch_before = time.perf_counter()
            batch_loss = 0.
            for p in params:
                p.grad = 0.
            for im in batch:
                set_input(im.label, im.pixels)
                for node in topo:
                    node._forward(node)
                for p in non_params:
                    p.grad = 0.
                loss.grad = 1
                for node in reverse_topo:
                    node._backward(node)
                batch_loss += loss.data
                epoch_loss += loss.data
            for p in params:
                p.data -= 0.1 * p.grad/batch_size
            batch_after = time.perf_counter()
            if batch_idx % 500 == 0:
                print(f"batch {batch_idx:4d} loss {batch_loss/batch_size:.2f} ({batch_size/(batch_after-batch_before):.2f} images/sec)")
        after = time.perf_counter()
        delta = after - before
        print(f"...epoch {epoch:4d} loss {epoch_loss/len(db):.2f} (took {delta:.2f} sec)")


run()
