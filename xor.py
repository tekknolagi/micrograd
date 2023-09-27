import random
from micrograd.nn import MLP
random.seed(4)
model = MLP(2, [5, 1])
batch = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
batch_size = len(batch)
for epoch in range(1000):
    model.zero_grad()
    epoch_loss = 0.
    outputs = [model(xs) for xs, _ in batch]
    expected = [exp for _, exp in batch]
    losses = [(exp-act)**2 for exp, act in zip(expected, outputs)]
    loss = sum(losses) * (1.0 / len(losses))
    loss.backward()
    for p in model.parameters():
        p.data -= 0.1 * p.grad / batch_size
    if epoch % 100 == 0:
        print(f"...epoch {epoch:4d} loss {loss.data/batch_size:.4f}")


print("params", [p.data for p in model.parameters()])
for im in batch:
    result = model(im[0])
    assert abs(result.data-im[1]) < 0.1, f"{result.data:.4f} is not near {im[1]}"
