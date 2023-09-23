import math
#import cython

counter = 0#cython.declare(cython.int)

def do_nothing(out): pass
def _backward_add(out):
    self, other = out._prev
    self.grad += out.grad
    other.grad += out.grad
def _backward_mul(out):
    self, other = out._prev
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad
def _backward_pow(out):
    self, = out._prev
    other = int(out._op[2:])
    self.grad += (other * self.data**(other-1)) * out.grad
def _backward_relu(out):
    self, = out._prev
    self.grad += (out.data > 0) * out.grad
def _backward_exp(out):
    self, = out._prev
    self.grad += math.exp(self.data) * out.grad
def _backward_log(out):
    self, = out._prev
    self.grad += 1/self.data * out.grad
def _forward_add(out):
    left, right = out._prev
    out.data = left.data + right.data
def _forward_mul(out):
    left, right = out._prev
    out.data = left.data * right.data
def _forward_pow(out):
    left, = out._prev
    other = int(out._op[2:])
    out.data = left.data ** other
def _forward_relu(out):
    left, = out._prev
    out.data = 0 if left.data < 0 else left.data
def _forward_exp(out):
    left, = out._prev
    out.data = math.exp(left.data)
def _forward_log(out):
    left, = out._prev
    out.data = math.log(left.data)


class Value:
    # data = cython.declare(cython.double, visibility='public')
    # grad = cython.declare(cython.double, visibility='public')
    # _backward = cython.declare(object, visibility='public')
    # _prev = cython.declare(tuple, visibility='readonly')
    # _op = cython.declare(str, visibility='readonly')
    # _id = cython.declare(cython.int, visibility='readonly')

    """ stores a single scalar value and its gradient """
    __slots__ = ("data", "grad", "_backward", "_forward", "_prev", "_op", "_id")

    def __init__(self, data, _children: tuple=(), _op:str=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = do_nothing
        self._forward = do_nothing
        self._prev = _children
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        # global counter
        # self._id = counter
        # counter += 1

    def __add__(self: "Value", other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        out._backward = _backward_add
        out._forward = _forward_add
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        out._backward = _backward_mul
        out._forward = _forward_mul
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')
        out._backward = _backward_pow
        out._forward = _forward_pow
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        out._backward = _backward_relu
        out._forward = _forward_relu
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        out._backward = _backward_exp
        out._forward = _forward_exp
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        out._backward = _backward_log
        out._forward = _forward_log
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
            v._backward(v)

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
        def _backward(self):
            if left.data > right.data:
                left.grad += self.grad
            else:
                right.grad += self.grad
        self._backward = _backward
        def _forward(self):
            self.data = max(left.data, right.data)
        self._forward = _forward


def dot(l, r):
    return sum(li.data*ri.data for li,ri in zip(l,r))


class Dot(Value):
    def __init__(self, left_arr, right_arr):
        assert len(left_arr) == len(right_arr)
        assert left_arr
        super().__init__(dot(left_arr, right_arr), tuple(set(left_arr+right_arr)), 'dot')
        self.left_arr = left_arr
        self.right_arr = right_arr
        def _forward(self):
            self.data = dot(self.left_arr, self.right_arr)
        self._forward = _forward
        def _backward(self):
            for i in range(len(self.left_arr)):
                self.left_arr[i].grad += self.right_arr[i].data*self.grad
                self.right_arr[i].grad += self.left_arr[i].data*self.grad
        self._backward = _backward
