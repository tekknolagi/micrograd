import torch
import numpy as np
from micrograd.vnn import Tensor as VTensor
from torch import Tensor as TTensor


def test_add():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[7, 8, 9], [10, 11, 12]])
    x0 = VTensor(x)
    y0 = VTensor(y)
    z0 = x0 + y0
    assert np.array_equal(x0.grad, [[0, 0, 0], [0, 0, 0]])
    assert np.array_equal(z0.data, [[8, 10, 12], [14, 16, 18]])

    x1 = TTensor(x)
    x1.requires_grad = True
    y1 = TTensor(y)
    y1.requires_grad = True
    z1 = x1 + y1
    z1.retain_grad()
    assert np.array_equal(z1.data, [[8, 10, 12], [14, 16, 18]])
    assert np.array_equal(z0.data, z1.data)

    z0.backward()
    z1.backward(TTensor(np.ones_like(z1.data)))
    assert np.array_equal(x0.grad, [[1, 1, 1], [1, 1, 1]])
    assert np.array_equal(x0.grad, x1.grad)
    assert np.array_equal(y0.grad, y1.grad)
    assert np.array_equal(z0.grad, z1.grad)


def test_sub():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[7, 8, 9], [10, 11, 12]])
    x0 = VTensor(x)
    y0 = VTensor(y)
    z0 = x0 - y0
    assert np.array_equal(x0.grad, [[0, 0, 0], [0, 0, 0]])
    assert np.array_equal(z0.data, [[-6, -6, -6], [-6, -6, -6]])

    x1 = TTensor(x)
    x1.requires_grad = True
    y1 = TTensor(y)
    y1.requires_grad = True
    z1 = x1 - y1
    z1.retain_grad()
    assert np.array_equal(z1.data, [[-6, -6, -6], [-6, -6, -6]])
    assert np.array_equal(z0.data, z1.data)

    z0.backward()
    z1.backward(TTensor(np.ones_like(z1.data)))
    assert np.array_equal(x0.grad, [[1, 1, 1], [1, 1, 1]])
    assert np.array_equal(x1.grad, [[1, 1, 1], [1, 1, 1]])
    assert np.array_equal(x0.grad, x1.grad)
    assert np.array_equal(y0.grad, y1.grad)
    assert np.array_equal(z0.grad, z1.grad)


def test_matmul():
    x = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    y = np.expand_dims(np.array([7, 8, 9]), axis=1)  # (3, 1)
    x0 = VTensor(x)
    y0 = VTensor(y)
    z0 = x0 @ y0
    assert np.array_equal(x0.grad, [[0, 0, 0], [0, 0, 0]])
    assert np.array_equal(z0.data, [[50], [122]])

    x1 = TTensor(x)
    x1.requires_grad = True
    y1 = TTensor(y)
    y1.requires_grad = True
    z1 = x1 @ y1
    z1.retain_grad()
    assert np.array_equal(z1.data, [[50], [122]])
    assert np.array_equal(z0.data, z1.data)

    z0.backward()
    z1.backward(TTensor(np.ones_like(z1.data)))
    assert np.array_equal(x0.grad, [[7, 8, 9], [7, 8, 9]])
    assert np.array_equal(x0.grad, x1.grad)
    assert np.array_equal(y0.grad, y1.grad)
    assert np.array_equal(z0.grad, z1.grad)
