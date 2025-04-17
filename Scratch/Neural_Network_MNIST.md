# Building Neural Network MNIST from Scratch

## Initialization (`__init__`)

This method sets up the neural network by initializing weights and biases for each layer:

- `W1`, `b1`: Weights and biases between the **input layer (784 neurons)** and the **first hidden layer (128 neurons)**.
- `W2`, `b2`: Between the **first hidden layer** and the **second hidden layer (64 neurons)**.
- `W3`, `b3`: Between the **second hidden layer** and the **output layer (10 neurons)**.
- Weights are initialized with random values scaled by He initialization: `np.sqrt(2. / previous_layer_size)`, suitable for ReLU activations.
- Biases are initialized as zeros.
- `lr`: Learning rate is stored for use in gradient descent.

## ReLU Activation (`relu`)

Applies ReLU activation function:

```python
return np.maximum(0, Z)
```

This introduces non-linearity and keeps positive values, zeroing out negatives.

## ReLU Derivative (`relu_deriv`)

Used during backpropagation to propagate gradients only where the activation was positive.

## Softmax Activation (`softmax`)

Turns logits from the output layer into probabilities:

```python
expZ = np.exp(Z - np.max(Z))
return expZ / np.sum(expZ)
```

- Numerically stable due to `- np.max(Z)`
- Used in classification tasks with one-hot targets

## Forward Pass (`forward`)

Feeds input data `X` through the network:

1. `Z1 = X dot W1 + b1` → then ReLU → `A1`
2. `Z2 = A1 dot W2 + b2` → then ReLU → `A2`
3. `Z3 = A2 dot W3 + b3` → then softmax → `A3` (final predictions)

## Loss Function (`compute_loss`)

Cross-entropy loss:

```python
loss = -np.sum(Y * np.log(Y_hat + 1e-8)) / m
```

- Measures the difference between predictions and actual labels
- `1e-8` avoids log(0)

## Backward Pass (`backward`)

Applies backpropagation to compute gradients and update weights:

1. `dZ3 = (A3 - Y) / m`: Gradient of softmax + loss
2. Compute `dW3`, `db3`, then propagate to:
3. `dA2`, `dZ2 = dA2 * relu_deriv(Z2)`
4. Compute `dW2`, `db2`, continue back to:
5. `dA1`, `dZ1 = dA1 * relu_deriv(Z1)`
6. Compute `dW1`, `db1`

Then update parameters with gradient descent:

```python
W -= lr * dW
b -= lr * db
```

## Prediction (`predict`)

Returns predicted class and full probabilities:

```python
np.argmax(probs, axis=1)
```

- `argmax`: gives the predicted digit
- `probs`: softmax output


