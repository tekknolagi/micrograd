from micrograd.nn import MLP
model = MLP(2, [4, 1])
batch = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
batch_size = len(batch)
for epoch in range(10):
    model.zero_grad()
    epoch_loss = 0.
    for xs, expected in batch:
        output = model(xs)
        loss = (output-expected)**2
        epoch_loss += loss.data
        loss.backward()
    for p in model.parameters():
        p.data -= 0.1 * p.grad / batch_size
    epoch_loss /= batch_size
    print(f"...epoch {epoch:4d} loss {epoch_loss:.4f}")

x = model([0,1])
print(len(x.topo()))

for im in batch:
    print(model(im[0]))


def mse(exp, act):
    return sum((ei - ai) ** 2 for ei, ai in zip(exp, act)) / len(exp)


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


