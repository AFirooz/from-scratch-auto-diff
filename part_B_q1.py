import numpy as np
import matplotlib.pyplot as plt
from micrograd.comp_graph import draw_dot

from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP


X_T = [[0.1, 0.7, 0.8, 0.8, 1.0, 0.3, 0.0, -0.3, -0.5, -1.5],
       [1.2, 1.8, 1.6, 0.6, 0.8, 0.5, 0.2, 0.8, -1.5, -1.3]]

y_T = [[1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]


def T(x:list[list]):
    """Transpose a matrix represented as a list of lists."""
    tx = []
    for jcol in range(len(x[0])):
        row = []
        for irow in range(len(x)):
            row.append(x[irow][jcol])
        tx.append(row)
    return tx

def shape(x):
    return (len(x), len(x[0])) if x else (0, 0)

X = T(X_T)
y = T(y_T)

print("X shape:", shape(X))
print("y shape:", shape(y))

model = MLP(2, [6, 2], nonlin='sigmoid') # 2-layer neural network
print("number of parameters", len(model.parameters()))
print(model)

t = model(X[0])
t[0].debug()


# def loss():
#     preds = []
#     for row in X:
#         preds.append(model(row))
#     return preds

# def loss():
#     preds = list(map(model, X))
#     losses = [(preds[i][0] - y[i][0])**2 + (preds[i][1] - y[i][1])**2 for i in range(len(y))]
#     data_loss = sum(losses) * (1.0 / len(losses))
#     return data_loss

# total_loss = loss()
# print(total_loss)
