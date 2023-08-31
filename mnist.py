import itertools
import random
import struct
import time
from micrograd.nn import MLP

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


def expected_digit(n):
    result = [0] * 10
    result[n] = 1.0
    return result


EXPECTED_DIGITS = [expected_digit(n) for n in range(10)]


def loss(label, output):
    # each output is a 10-vector for what digit it is likely to be
    expected = EXPECTED_DIGITS[label]
    # mean squared error (mse)
    return sum((exp - act) ** 2 for exp, act in zip(expected, output))


def argmax(output):
    return max(range(len(output)), key=output.__getitem__)


# class Loss:
#     def compile(self):
#         raise NotImplementedError
# 
# 
# class MSE(Loss):
#     def compile(self):
#         result = []
#         result.append(
#             f"INLINE double {self.func_name()}(const Vector<double, {len(self.w)}>& input) {{",
#         )
#         result.append(
#             "double result = "
#             + " + ".join(
#                 f"{wi.data}*input.at({xi})" for xi, wi in enumerate(self.w)
#             )
#             + f" + {self.b.data};"
#         )
#         if self.nonlin:
#             # relu
#             result.append("result = std::max(result, double{0});")
#         result.append("return result;")
#         result.append("}")
#         return result


print("Making model...")
model = MLP(PIXEL_LENGTH, [512, 10])
print("Opening images...")
db = images("train-images-idx3-ubyte", "train-labels-idx1-ubyte")

TRAINING_ROUNDS = 50
IMAGES_PER_ROUND = db.num_images // TRAINING_ROUNDS
for k in range(TRAINING_ROUNDS):
    # forward
    print(f"Step {k}...", end=" ")
    before = time.perf_counter()
    inputs = list(itertools.islice(db, IMAGES_PER_ROUND))
    labels = [image.label for image in inputs]
    outputs = [model(image.pixels) for image in inputs]
    losses = [loss(label, output) for label, output in zip(labels, outputs)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    # alpha = 1e-4
    # reg_loss = alpha * sum((p * p for p in model.parameters()))
    reg_loss = 0
    total_loss = data_loss + reg_loss
    # also get accuracy
    acc = [yi == argmax(scorei) for yi, scorei in zip(labels, outputs)]
    acc = sum(acc)/len(acc)
    after = time.perf_counter()
    delta = after - before
    print(f"loss {total_loss.data:.2f}, accuracy {acc*100:.2f}% ({delta:.2f} seconds)")

    # backward
    before = time.perf_counter()
    model.zero_grad()
    total_loss.backward()
    after = time.perf_counter()
    delta = after - before
    print(f"  (backprop took {delta:.2f} seconds)")

    # update (sgd)
    before = time.perf_counter()
    learning_rate = 1.0 - 0.9 * k / 100
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    after = time.perf_counter()
    delta = after - before
    print(f"  (update took {delta:.2f} seconds)")
