import math

class Value:
    """ stores a single scalar value and its gradient """
    __slots__ = ("data", "grad", "_backward", "_forward", "_prev", "_op")

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._forward = lambda: None
        self._prev = _children
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        def _forward():
            out.data = self.data + other.data
        out._forward = _forward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        def _forward():
            out.data = self.data * other.data
        out._forward = _forward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        def _forward():
            out.data = self.data ** other
        out._forward = _forward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        def _forward():
            out.data = 0 if self.data < 0 else self.data
        out._forward = _forward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += math.exp(self.data) * out.grad
        out._backward = _backward
        def _forward():
            out.data = math.exp(self.data)
        out._forward = _forward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += 1/self.data * out.grad
        out._backward = _backward
        def _forward():
            out.data = math.log(self.data)
        out._forward = _forward
        return out

    def topo(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        return topo

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


class Max(Value):
    def __init__(self, left, right):
        super().__init__(max(left.data, right.data), (left, right), 'max')
        def _backward():
            if left.data > right.data:
                left.grad += self.grad
            else:
                right.grad += self.grad
        self._backward = _backward
        def _forward():
            self.data = max(left.data, right.data)
        self._forward = _forward


def dot(l, r):
    return sum(li.data*ri.data for li,ri in zip(l,r))


class Dot(Value):
    def __init__(self, left_arr, right_arr):
        assert len(left_arr) == len(right_arr)
        assert left_arr
        super().__init__(dot(left_arr, right_arr), tuple(set(left_arr+right_arr)), 'dot')
        def _forward():
            self.data = dot(left_arr, right_arr)
        self._forward = _forward
        def _backward():
            for li, ri in zip(left_arr, right_arr):
                li.grad += ri.data*self.grad
                ri.grad += li.data*self.grad
        self._backward = _backward
