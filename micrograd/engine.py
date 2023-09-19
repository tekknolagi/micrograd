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

    def var(self):
        if self._op == '':
            return str(self.data)
        return f"data[{self._id}]"

    def set(self, val):
        if self._op == '':
            raise RuntimeError("Can't set constant")
        return f"{self.var()} = {val};"

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
            return self.set(f"{self._prev[0].var()}*{self._prev[1].var()}")
        if self._op == '+':
            assert len(self._prev) == 2
            return self.set(f"{self._prev[0].var()}+{self._prev[1].var()}")
        if self._op == 'ReLU':
            assert len(self._prev) == 1
            return self.set(f"relu({self._prev[0].var()})")
        if self._op.startswith('**'):
            exponent = int(self._op[2:])
            assert len(self._prev) == 1
            return self.set(self.make_exp(self._prev[0].var(), exponent))
        if self._op == 'exp':
            return self.set(f"exp({self._prev[0].var()})")
        if self._op == 'log':
            return self.set(f"log({self._prev[0].var()})")
        raise NotImplementedError(self._op)

    def getgrad(self):
        if self._op in ('', 'input'):
            raise RuntimeError("Grad for constants and input data not stored")
        return f"grad[{self._id}]"

    def setgrad(self, val):
        if self._op in ('', 'input'):
            return []
        return [f"{self.getgrad()} += clip({val});"]

    def backward_compile(self):
        if not self._prev:
            assert self._op in ('', 'weight', 'bias', 'input')
            # Nothing to propagate to children.
            assert not self._prev
            return []
        if self._op == '*':
            left, right = self._prev
            return left.setgrad(f"{right.var()}*{self.getgrad()}") +\
                    right.setgrad(f"{left.var()}*{self.getgrad()}")
        if self._op == '+':
            left, right = self._prev
            return left.setgrad(f"{self.getgrad()}") + right.setgrad(f"{self.getgrad()}")
        if self._op == 'ReLU':
            prev, = self._prev
            return prev.setgrad(f"({self.var()}>0)*{self.getgrad()}")
        if self._op.startswith('**'):
            exponent = int(self._op[2:])
            prev, = self._prev
            exp = self.make_exp(prev.var(), exponent-1)
            return prev.setgrad(f"{exponent}*{exp}*{self.getgrad()}")
        if self._op == 'exp':
            prev, = self._prev
            return prev.setgrad(f"exp({prev.var()})*{self.getgrad()}")
        if self._op == 'log':
            prev, = self._prev
            return prev.setgrad(f"1.0L/{prev.var()}*{self.getgrad()}")
        raise NotImplementedError(self._op)

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
        return self.set(f"fmax({left.var()}, {right.var()})")

    def backward_compile(self):
        left, right = self._prev
        return [f"if ({left.var()} > {right.var()}) {{"] +\
                left.setgrad(f"{self.getgrad()}") +\
                ["} else {"] +\
                right.setgrad(f"{self.getgrad()}") +\
                ["}"]
