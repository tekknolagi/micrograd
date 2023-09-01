import itertools
import micrograd
import torch
from micrograd.engine import Value

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

def test_topo():
    a = Value(1)
    b = Value(2)
    c = a + b
    topo = c.topo()
    assert topo == [a, b, c]

def test_topo_bigger():
    a = Value(2)
    b = Value(3)
    c = Value(4)
    d = Value(5)
    ab = a * b
    cd = c * d
    e = ab + cd
    topo = e.topo()
    assert topo == [a, b, ab, c, d, cd, e]

def test_compile_value():
    a = Value(123)
    assert a.compile() == f"data[{a._id}] = 123;"

def test_compile_add():
    a = Value(0)
    b = Value(0)
    c = a + b
    assert c.compile() == f"data[{c._id}] = data[{a._id}]+data[{b._id}];"

def test_compile_mul():
    a = Value(0)
    b = Value(0)
    c = a * b
    assert c.compile() == f"data[{c._id}] = data[{a._id}]*data[{b._id}];"

def test_compile_add_topo():
    a = Value(1)
    b = Value(2)
    c = a + b
    topo = c.topo()
    result = "\n".join(t.compile() for t in topo)
    assert result == f"""\
data[{a._id}] = 1;
data[{b._id}] = 2;
data[{c._id}] = data[{a._id}]+data[{b._id}];\
"""

def test_commpile_addmul_topo():
    a = Value(2)
    b = Value(3)
    c = Value(4)
    d = Value(5)
    ab = a * b
    cd = c * d
    e = ab + cd
    topo = e.topo()
    result = "\n".join(t.compile() for t in topo)
    assert result == f"""\
data[{a._id}] = 2;
data[{b._id}] = 3;
data[{ab._id}] = data[{a._id}]*data[{b._id}];
data[{c._id}] = 4;
data[{d._id}] = 5;
data[{cd._id}] = data[{c._id}]*data[{d._id}];
data[{e._id}] = data[{ab._id}]+data[{cd._id}];\
"""

def test_backward_compile_add():
    a = Value(0)
    b = Value(0)
    c = a + b
    assert c.backward_compile() == [
        f"grad[{a._id}] += grad[{c._id}];",
        f"grad[{b._id}] += grad[{c._id}];",
    ]

def test_backward_compile_mul():
    a = Value(0)
    b = Value(0)
    c = a * b
    assert c.backward_compile() == [
        f"grad[{a._id}] += data[{b._id}]*grad[{c._id}];",
        f"grad[{b._id}] += data[{a._id}]*grad[{c._id}];",
    ]
