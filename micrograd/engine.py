counter = 0

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self.forwarded = None
        global counter
        self._id = counter
        counter += 1

    def find(self):
        op = self
        while isinstance(op, Value):
            next = op.forwarded
            if next is None:
                return op
            op = next
        return op

    def args(self):
        return tuple(v.find() for v in self._prev)

    def make_equal_to(self, other):
        self.find().set_forwarded(other)

    def set_forwarded(self, other):
        if self._op == '':
            assert self.data == other.data
        else:
            self.forwarded = other

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def topo(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.args():
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        return topo

    def backward(self):

        topo = self.topo()
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


class Dot(Value):
    def __init__(self, left, right):
        super().__init__(0, (), 'dot')
        assert len(left) == len(right)
        self._left = left
        self._right = right
        self._prev = tuple(left+right)
        global counter
        self._id = counter
        counter += 1

        # TODO(max): Figure out a way to compute this automatically using chain
        # rule.
        def _backward():
            for i in range(len(self._left)):
                self._left[i].grad += self._right[i].data*self.grad
                self._right[i].grad += self._left[i].data*self.grad

        self._backward = _backward

    def __repr__(self):
        return f"Dot(left={self._left}, right={self._right})"
