import math
import functools


def do_nothing(self, *args):
    pass


def backward_mul(out, self, other):
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad


def backward_pow(out, self):
    other = float(out._op[2:])
    self.grad += (other * self.data**(other-1)) * out.grad


def backward_add(out, self, other):
    self.grad += out.grad
    other.grad += out.grad


def backward_relu(out, self):
    self.grad += (out.data > 0) * out.grad


counter = 0


class Value:
    """ stores a single scalar value and its gradient """
    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "_id")

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = do_nothing
        self._prev = _children
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        global counter
        self._id = counter
        counter += 1

    def var(self):
        return f"data[{self._id}]"

    def set(self, val):
        return f"{self.var()} = {val};"

    def compile(self):
        if self._op == '':
            return self.set(self.data)
        if self._op in ('weight', 'bias', 'input'):
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
            return self.set(f"pow({self._prev[0].var()}, {exponent})")
        if self._op == 'exp':
            return self.set(f"exp({self._prev[0].var()})")
        if self._op == 'log':
            return self.set(f"log({self._prev[0].var()})")
        raise NotImplementedError(self._op)

    def getgrad(self):
        return f"grad[{self._id}]"

    def backward_compile(self):
        if not self._prev:
            assert self._op in ('', 'weight', 'bias', 'input')
            # Nothing to propagate to children.
            assert not self._prev
            return []
        if self._op == '*':
            left, right = self._prev
            return [
                f"{left.getgrad()} += {right.var()}*{self.getgrad()};",
                f"{right.getgrad()} += {left.var()}*{self.getgrad()};",
                ]
        if self._op == '+':
            left, right = self._prev
            return [
                f"{left.getgrad()} += {self.getgrad()};",
                f"{right.getgrad()} += {self.getgrad()};",
                ]
        if self._op == 'ReLU':
            prev, = self._prev
            return [f"{prev.getgrad()} += ({self.var()}>0)*{self.getgrad()};"]
        if self._op.startswith('**'):
            exponent = float(self._op[2:])
            prev, = self._prev
            return [f"{prev.getgrad()} += {exponent}*pow({prev.var()}, {exponent-1})*{self.getgrad()};"]
        if self._op == 'exp':
            prev, = self._prev
            return [f"{prev.getgrad()} += exp({prev.var()})*{self.getgrad()};"]
        if self._op == 'log':
            prev, = self._prev
            return [f"{prev.getgrad()} += 1.0L/{prev.var()}*{self.getgrad()};"]
        raise NotImplementedError(self._op)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, (), '')
        out = Value(self.data + other.data, (self, other), '+')
        out._backward = backward_add
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, (), '')
        out = Value(self.data * other.data, (self, other), '*')
        out._backward = backward_mul
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')
        out._backward = backward_pow
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        out._backward = backward_relu
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward(out, self):
            self.grad += 1/self.data * out.grad
        out._backward = _backward
        return out

    def topo(self):
        seen = set()
        stack = []    # path variable is gone, stack and order are new
        order = []    # order will be in reverse order at first
        q = [self]
        while q:
            v = q.pop()
            if v not in seen:
                seen.add(v) # no need to append to path any more
                q.extend(v._prev)

                while stack and v not in stack[-1]._prev: # new stuff here!
                    order.append(stack.pop())
                stack.append(v)

        return list(reversed(stack + order[::-1]))   # new return value!


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
            # else:
            #     assert v._op in ('', 'weight', 'bias', 'input'), f"uh oh {v._op}"
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward(v, *v._prev)

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward(out, self):
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
    def __init__(self, values):
        super().__init__(max(v.data for v in values), tuple(values), 'max')

    def compile(self):
        prev = (v.var() for v in self._prev)
        val = functools.reduce(lambda x, y: f"fmax({x}, {y})", prev)
        return self.set(val)

    def backward_compile(self):
        prev_id = (str(v._id) for v in self._prev)
        return [
                f"{{ double input{self._id}[] = {{ {', '.join(prev_id)} }};",
                f"int idx{self._id} = argmax({len(self._prev)}, input{self._id});",
                f"grad[idx{self._id}] += {self.getgrad()}; }}",
                ]
