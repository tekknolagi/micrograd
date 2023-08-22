import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

    def func_name(self):
        return f"{self.__class__.__name__}_{hex(id(self))}"


class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1), (), 'weight') for _ in range(nin)]
        self.b = Value(0, (), 'bias')
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def sig(self, params):
        return f"double {self.func_name()}(Vector<double, {len(params)}> input)"

    def compile(self, x):
        result = [
                self.sig(x) + " {",
                "double result = 0;",
                ]
        weights = ", ".join(str(wi.data) for wi in self.w)
        result.append(f"Vector<double, {len(self.w)}> weights = {{ {weights} }};")
        result.append(f"double result = weights.dot(input).sum() + {self.b.data};")
        # for idx, (wi, xi) in enumerate(zip(self.w, x)):
        #     result.append(f"result += {wi.data}*input.at({idx});")
        # result.append(f"result += {self.b.data};")
        if self.nonlin:
            # relu
            result.append("result = std::max(result, 0);")
        result.append("return result;")
        result.append("}")
        return result

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def sig(self, params):
        dim = len(self.neurons)
        return f"Vector<double, {dim}> {self.func_name()}(Vector<double, {len(params)}> input)"

    def compile(self, x):
        result = []
        for n in self.neurons:
            result += n.compile(x)
        dim = len(self.neurons)
        result += [
                self.sig(x) + " {",
                f"Vector<double, {dim}> result[{dim}];",
                ]
        for idx, n in enumerate(self.neurons):
            result.append(f"result.at({idx}) = {n.func_name()}(input);")
        result.append("return result;")
        result.append("}")
        return result

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def sig(self, params):
        dim_in = len(self.layers[0].neurons)
        dim_out = len(self.layers[-1].neurons)
        return f"Vector<double, {dim_out}> {self.func_name()}(Vector<double, {dim_in}> input)"

    def compile(self, x):
        result = []
        for layer in self.layers:
            result += layer.compile(x)
        result += [
                self.sig(x) + " {",
                ]
        for idx, layer in enumerate(self.layers):
            dim_out = len(layer.neurons)
            inp = "input" if idx == 0 else f"result{idx-1}"
            result.append(f"Vector<double, {dim_out}> result{idx} = {layer.func_name()}({inp});")
        dim_out = len(self.layers[-1].neurons)
        result.append(f"return result{len(self.layers)-1};")
        result.append("}")
        return result

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
