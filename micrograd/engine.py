import math

class Value:
    """ Stores a single scalar value and its gradient.
    This class is used to represent a value in a computational graph and use that to calculate gradients computationally.
    """

    def __init__(self, data, _children=(), _op='', label:str=None):
        """
        Notes (_backward):
            - The `_backward` now will be a function that will define how the downstream gradient have to be distributed to the children (chained with local gradient) to get the upstream gradient.
            
            - by default, we have no operation to perform and no gradient to propagate, so return None. Each mathmatical operation (e.g. `__add__()`) will override this to define its own backward function.

            - Also note that any function defined must add to the old gradient (self.grad) to allow for multiple paths to the same node. In other words, when a node is used multiple times in the graph, we need to accumulate the gradients.

        Notes (_prev):
            - why use a set?
            Because we will not need to have duplicates since we will deal with that in the graph creation function trace().
        """

        self.data = data
        self.label = label
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the operation that produced this node, for graphviz / debugging / etc

    def __add__(self, other): # self + other
        # to be able to add a number to another Value(number)
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        # Define how to distribute the gradient for this operation. Remember `_backward` must be a function
        def _backward():
            # Note that we are just passing the out.grad (downstream grad) to the children
            # The local gradient is 1 for both children since out = self + other
            # so d(out)/d(self) = 1 and d(out)/d(other) = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward # update the self._backward function for this operation

        return out

    def __mul__(self, other): # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        # see _backward implementation in __add__() for details
        def _backward():
            # out = self * other
            # downstream gradient: out.grad
            # local grad: d(out)/d(self) = other
            self.grad += other.data * out.grad
            # local grad: d(out)/d(other) = self
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

    def sigmoid(self):
        out = Value(1 / (1 + math.exp(-self.data)), _children=(self,), _op='Sigmoid')

        # see _backward implementation in __add__() and __mul__() for details
        def _backward():
            """
            out = sigmoid(self)
            local gradient: d(out)/d(self) = out * (1 - out)
            downstream gradient: out.grad
            """
            self.grad += out.data * (1 - out.data) * out.grad
        
        out._backward = _backward
        return out

    def _topo_sort(self):
        """
        A topological sort (or order) is a graph traversal in which each node v is visited only after all its dependencies are visited.
        This is needed for implementing the backward() method.
        We will reverse this list when calling backward(). This way, we ensure that all the parent _backward() are called before the childeren.
        """
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
        """ Perform backpropagation to compute gradients.
        In other words, it will traverse the graph in reverse order (parents first) and call _backwards() to compute the gradients.
        
        Note that the root's gradient needs to be initialized to 1, since we are computing the gradient of the output with respect to itself. Otherwise, since it is zero by default, all children will use zero as the downstream gradient.
        """
        topo = self._topo_sort()
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

    def debug(self)->str:
        """ Use to view the computational graph in text mode. """
        data_grad = f"data={self.data:.3f}, grad={self.grad:.3f}"
        data_label = f"{self.label}" if self.label is not None else "Value"
        
        # If no children, just return the data and grad info
        if len(self._prev) == 0:
            return f"{data_label}({data_grad})\n"
        operation = f"operation='{self._op}'"
        
        # Format with fixed-width fields
        txt = f"{data_label}({data_grad}, {operation}):\n"
        
        # Add children indented and aligned
        for child in self._prev:
            child_lines = child.debug().split('\n')
            for j, line in enumerate(child_lines):
                if line:
                    # Add two spaces of indentation per level plus a vertical pipe
                    txt += f"  │ {line}\n"
        return txt

    def __repr__(self):
        """ Used when displaying the object """
        txt = f"Value(data={self.data}, grad={self.grad})" if self.label is None \
            else f"{self.label}(data={self.data}, grad={self.grad})"
        return txt
