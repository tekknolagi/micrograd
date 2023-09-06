import torch
from micrograd.engine import Value
from micrograd.nn import MLP

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol


def test_duplicate_backprop():
    a = Value(1.0)
    b = a + 4
    c = (b*3)+(b*5)
    c.backward()
    assert a.grad == 8.
    assert b.grad == 8.
    assert c.grad == 1.


    a = torch.tensor([1.], requires_grad=True)
    b = a + 4; b.retain_grad()
    c = (b*3)+(b*5); c.retain_grad()
    c.backward()
    assert a.grad.item() == 8.
    assert b.grad.item() == 8.
    assert c.grad.item() == 1.


def test_inplace_forward():
    a = Value(1.0)
    b = Value(2.0)
    c = Value(3.0)
    c = a + b * c
    assert c.data == 7.

    a.data = 8
    c.forward()
    assert c.data == 14.


def test_inplace_forward_mlp():
    model = MLP(2, [16, 1])
    expected = model([Value(3), Value(4)])

    x = Value(0)
    y = Value(0)
    actual = model([x, y])
    x.data = 3
    y.data = 4
    actual.forward()

    assert actual.data == expected.data
