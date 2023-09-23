import math


counter = 0


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = _children
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        global counter
        self._id = counter
        counter += 1

    def getdata(self):
        if self._op == '':
            return str(self.data)
        return f"data[{self._id}]"

    def set(self, val):
        if self._op == '':
            raise RuntimeError("Can't set constant")
        return f"{self.getdata()} = {val};"

    def make_exp(self, val, exp):
        if exp == 0:
            return "1"
        elif exp == 1:
            return val
        elif exp == -1:
            return f"((double)1)/{val}"
        return f"pow({val}, {exp})"

    def compile(self):
        if self._op in ('', 'weight', 'bias', 'input'):
            # Set once at init time and thereafter reset in update
            return ""
        if self._op == '*':
            assert len(self._prev) == 2
            return self.set(f"{self._prev[0].getdata()}*{self._prev[1].getdata()}")
        if self._op == '+':
            assert len(self._prev) == 2
            return self.set(f"{self._prev[0].getdata()}+{self._prev[1].getdata()}")
        if self._op == 'ReLU':
            assert len(self._prev) == 1
            return self.set(f"relu({self._prev[0].getdata()})")
        if self._op.startswith('**'):
            exponent = int(self._op[2:])
            assert len(self._prev) == 1
            return self.set(self.make_exp(self._prev[0].getdata(), exponent))
        if self._op == 'exp':
            return self.set(f"exp({self._prev[0].getdata()})")
        if self._op == 'log':
            return self.set(f"log({self._prev[0].getdata()})")
        raise NotImplementedError(self._op)

    def getgrad(self):
        if self._op in ('', 'input'):
            raise RuntimeError("Grad for constants and input data not stored")
        return f"grad[{self._id}]"

    def setgrad(self, val):
        if self._op in ('', 'input'):
            return []
        return [f"{self.getgrad()} += clip({val});"]

    def backward_compile(out):
        if not out._prev:
            assert out._op in ('', 'weight', 'bias', 'input')
            # Nothing to propagate to children.
            assert not out._prev
            return []
        if out._op == '*':
            self, other = out._prev
            return self.setgrad(f"{other.getdata()}*{out.getgrad()}") +\
                    other.setgrad(f"{self.getdata()}*{out.getgrad()}")
        if out._op == '+':
            self, other = out._prev
            return self.setgrad(f"{out.getgrad()}") + other.setgrad(f"{out.getgrad()}")
        if out._op == 'ReLU':
            self, = out._prev
            return self.setgrad(f"({out.getdata()}>0)*{out.getgrad()}")
        if out._op.startswith('**'):
            exponent = int(out._op[2:])
            self, = out._prev
            exp = out.make_exp(self.getdata(), exponent-1)
            return self.setgrad(f"{exponent}*{exp}*{out.getgrad()}")
        if out._op == 'exp':
            self, = out._prev
            return self.setgrad(f"exp({self.getdata()})*{out.getgrad()}")
        if out._op == 'log':
            self, = out._prev
            return self.setgrad(f"1.0L/{self.getdata()}*{out.getgrad()}")
        raise NotImplementedError(out._op)

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

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += 1/self.data * out.grad
        out._backward = _backward

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

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += math.exp(self.data) * out.grad
        out._backward = _backward

        return out

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

    def compile(self):
        left, right = self._prev
        return self.set(f"fmax({left.getdata()}, {right.getdata()})")

    def backward_compile(self):
        left, right = self._prev
        return [f"if ({left.getdata()} > {right.getdata()}) {{"] +\
                left.setgrad(f"{self.getgrad()}") +\
                ["} else {"] +\
                right.setgrad(f"{self.getgrad()}") +\
                ["}"]


def dot(l, r):
    return sum(li.data*ri.data for li,ri in zip(l,r))


class Dot(Value):
    def __init__(self, left_arr, right_arr):
        assert len(left_arr) == len(right_arr)
        assert left_arr
        super().__init__(dot(left_arr, right_arr), tuple(set(left_arr+right_arr)), 'dot')
        self.left_arr = left_arr
        self.right_arr = right_arr

    def compile(self):
        products = (f"{li.getdata()}*{ri.getdata()}" for li, ri in zip(self.left_arr, self.right_arr))
        return self.set(f"{'+'.join(products)}")

    def backward_compile(self):
        result = []
        for i in range(len(self.left_arr)):
            result += self.left_arr[i].setgrad(f"{self.right_arr[i].getdata()}*{self.getgrad()}")
            result += self.right_arr[i].setgrad(f"{self.left_arr[i].getdata()}*{self.getgrad()}")
        return result
