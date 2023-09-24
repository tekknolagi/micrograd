import random
from micrograd.nn import MLP
random.seed(1)
model = MLP(2, [4, 1])
batch = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
batch_size = len(batch)
for epoch in range(100):
    model.zero_grad()
    epoch_loss = 0.
    for xs, expected in batch:
        output = model(xs)
        loss = (output-expected)**2
        epoch_loss += loss.data
        loss.backward()
    for p in model.parameters():
        p.data -= 0.1 * p.grad / batch_size
    print(f"...epoch {epoch:4d} loss {epoch_loss/batch_size:.4f}")


for im in batch:
    result = model(im[0])
    assert abs(result.data-im[1]) < 0.1, f"{result.data:.4f} is not near {im[1]}"
