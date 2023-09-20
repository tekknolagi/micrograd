import math
import random
from micrograd.nn import MLP

random.seed(1337)


def mse(exp, act):
    return sum((ei - ai) ** 2 for ei, ai in zip(exp, act)) / len(exp)


model = MLP(2, [4, 1])
db = [
    (bytes([0, 0]), 0),
    (bytes([0, 1]), 1),
    (bytes([1, 0]), 1),
    (bytes([1, 1]), 0),
]
times = 1
batch = db*times
batch_size = len(batch)
num_epochs = 5


# for epoch in range(num_epochs):
#     # print("---epoch---")
#     output = (model(im[0]) for im in batch)
#     loss = mse([im[1] for im in batch], output)
#     model.zero_grad()
#     loss.backward()
#     assert not math.isinf(loss.data)
#     assert not math.isnan(loss.data)
#     for p in model.parameters():
#         p.data -= 0.1 * p.grad
#     if epoch % 200 == 0:
#         print(f"...epoch {epoch:4d} loss {loss.data:.4f}")


for epoch in range(num_epochs):
    model.zero_grad()
    epoch_loss = 0.
    for im in batch:
        output = model(im[0])
        print("exp", im[1], "out", output.data)
        loss = (output-im[1])**2
        # print([o._id for o in loss.topo()])
        # print("[","\n".join([str(o._id) for o in loss.topo()]),"]")
        # break
        epoch_loss += loss.data
        assert not math.isinf(loss.data)
        assert not math.isnan(loss.data)
        loss.backward()
        # for p in model.parameters():
        #     print("grad", p._id, p.grad)
    for p in model.parameters():
        p.data -= 0.1 * p.grad / batch_size
    epoch_loss /= batch_size
    print(f"...epoch {epoch:4d} loss {epoch_loss:.4f}")

for im in db:
    print(model(im[0]))
