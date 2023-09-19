import argparse
import functools
import itertools
import os
import random
import shutil
import struct
import sys
import tempfile
import time
from micrograd import nn as nn_interp
from micrograd.engine import Max
from tqdm import tqdm


tqdm.monitor_interval = 0
random.seed(1337)
sys.setrecursionlimit(20000)


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


NUM_DIGITS = 10
model = timer(lambda: nn_interp.MLP(DIM, [50, NUM_DIGITS]), "Building model...")


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


def stable_softmax(output):
    max_ = functools.reduce(Max, output)
    shiftx = [o-max_ for o in output]
    exps = [o.exp() for o in shiftx]
    sum_ = sum(exps)
    return [o/sum_ for o in exps]


def loss_of(model, image):
    output = model(image.pixels)
    softmax_output = stable_softmax(output)
    expected_onehot = [0. for _ in range(NUM_DIGITS)]
    expected_onehot[image.label] = 1.
    result = -sum(exp*(act+0.0001).log() for exp, act in zip(expected_onehot, softmax_output))
    return result


print("Training...")
num_epochs = 100
db = list(images("train-images-idx3-ubyte", "train-labels-idx1-ubyte"))
batch_size = 1000
for epoch in range(num_epochs):
    epoch_loss = 0.
    before = time.perf_counter()
    shuffled = db.copy()
    random.shuffle(shuffled)
    for batch_idx, batch in tqdm(enumerate(grouper(batch_size, shuffled))):
        for p in model.parameters():
            p.grad = 0.0
        num_correct = 0
        loss = sum(loss_of(model, im) for im in tqdm(batch))
        loss.backward()
        epoch_loss += loss.data
        for p in model.parameters():
            p.data -= 0.1 * p.grad
    after = time.perf_counter()
    delta = after - before
    epoch_loss /= len(db)
    print(f"...epoch {epoch:4d} loss {epoch_loss:.2f} (took {delta} sec)")
