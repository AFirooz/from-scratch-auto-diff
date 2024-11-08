import random
from micrograd.engine import Value

class Module:
    """Base class that holds common functionality."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """Return a list of parameters (weights and biases) of the module.
        By default, it returns an empty list. Subclasses should override this method.
        """
        return []

class Neuron(Module):
    """A single neuron with weights and bias."""

    def __init__(self, nin, nonlin:str=None):
        """
        If nonlin is None, the neuron will be linear. Otherwise, it will apply the specified nonlinearity using lamda functions.
        """
        self.w = [Value(random.uniform(-1,1), label=f'w{i}') for i in range(nin)]
        self.b = Value(0, label='b')

        if nonlin is not None:
            assert nonlin in ["relu", "sigmoid"], "Invalid nonlinearity. See Neuron.__init__()."
            self.str_nonlin = nonlin
            match nonlin:
                case "relu":
                    self.nonlin = lambda x: x.relu() # assuming x is a Value object and the relu method is defined in Value class.
                case "sigmoid":
                    self.nonlin = lambda x: x.sigmoid()
        else:
            self.nonlin = lambda x: x.linear()
            self.str_nonlin = 'linear'

    def __call__(self, x):
        assert len(x) == len(self.w), "X size must match weights size."
        z = sum((wi*xi for wi,xi in zip(self.w, x)), self.b); z.label = 'z'
        act = self.nonlin(z); act.label = 'act'
        return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.str_nonlin}Neuron(in:{len(self.w)})"

class Layer(Module):
    """A layer of neurons."""

    def __init__(self, nin:int, nout:int, **kwargs):
        """
        nin: number of inputs to each neuron
        nout: number of neurons in the layer
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons] # we just call each neuron and collect the outputs in a list
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()] # for n in self.neurons: for p in n.parameters(): collect p

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """Multi-layer perceptron
    nin: An integer representing the number of input neurons (or features) for the network.
    nouts: A list of integers, where each integer specifies the number of neurons in each successive hidden layer.
    """

    def __init__(self, nin:int, nouts:list, nonlin:str=None):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(sz)-1):
            if nonlin is None:
                nonlin = 'relu' if i != len(sz)-2 else None  # not applying nonlinearity to the last layer. Just calculating the logits.
            else:
                nonlin = nonlin if i != len(sz)-2 else None
            self.layers.append(Layer(sz[i], sz[i+1], nonlin=nonlin))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        n = ',\n' # to avoid SyntaxError of using backslash in f-string
        return f"MLP of [\n{n.join(str(layer) for layer in self.layers)}\n]"
