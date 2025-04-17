import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype(np.float32) / 255.0
y = y.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/7, random_state=42
)

def one_hot(y, num_classes):
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot

y_train_o = one_hot(y_train, 10)
y_test_o = one_hot(y_test, 10)

class NeuralNetwork:
    def __init__(self, input_dim, hidden1, hidden2, output_dim, lr=0.1, seed=1):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2. / hidden1)
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, output_dim) * np.sqrt(2. / hidden2)
        self.b3 = np.zeros((1, output_dim))
        self.lr = lr

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return (Z > 0).astype(np.float32)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        self.Z3 = self.A2.dot(self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)
        return self.A3

    def compute_loss(self, Y_hat, Y):
        m = Y.shape[0]
        loss = -np.sum(Y * np.log(Y_hat + 1e-8)) / m
        return loss

    def backward(self, X, Y):
        m = X.shape[0]
        dZ3 = (self.A3 - Y) / m
        dW3 = self.A2.T.dot(dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        dA2 = dZ3.dot(self.W3.T)
        dZ2 = dA2 * self.relu_deriv(self.Z2)
        dW2 = self.A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = X.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1), probs
    
input_dim = 784
hidden1 = 128
hidden2 = 64
output_dim = 10
nn = NeuralNetwork(input_dim, hidden1, hidden2, output_dim, lr=0.1)

epochs = 20
batch_size = 256
train_losses, train_accuracies = [], []

for epoch in range(epochs):
    perm = np.random.permutation(X_train.shape[0])
    X_shuff, Y_shuff = X_train[perm], y_train_o[perm]

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_shuff[i:i+batch_size]
        Y_batch = Y_shuff[i:i+batch_size]
        Y_hat = nn.forward(X_batch)
        nn.backward(X_batch, Y_batch)

    train_pred, _ = nn.predict(X_train)
    acc = np.mean(train_pred == y_train)
    loss = nn.compute_loss(nn.forward(X_train), y_train_o)
    train_accuracies.append(acc)
    train_losses.append(loss)
    print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f} - acc: {acc:.4f}")

fig, axes = plt.subplots(2, 4, figsize=(8, 4))
indices = np.random.choice(X_train.shape[0], 8, replace=False)
for ax, idx in zip(axes.flatten(), indices):
    ax.imshow(X_train[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {y_train[idx]}")
    ax.axis('off')
plt.suptitle("Random MNIST Training Samples")
plt.tight_layout()
plt.show()

sample_ids = np.random.choice(X_test.shape[0], 4, replace=False)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, idx in enumerate(sample_ids):
    img = X_test[idx].reshape(28, 28)
    pred, probs = nn.predict(X_test[idx:idx+1])
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title(f"True: {y_test[idx]}, Pred: {pred[0]}")
    axes[0, i].axis('off')
    axes[1, i].bar(range(10), probs.ravel())
    axes[1, i].set_xticks(range(10))
    axes[1, i].set_ylim(0, 1)
    axes[1, i].set_title("Class Probabilities")

plt.tight_layout()
plt.show()

test_pred, _ = nn.predict(X_test)
test_acc = np.mean(test_pred == y_test)
print(f"Test Accuracy: {test_acc:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(range(1, len(train_losses)+1), train_losses, marker='o')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax2.plot(range(1, len(train_accuracies)+1), train_accuracies, marker='o')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training Accuracy")
plt.tight_layout()
plt.show()

mis_idx = np.where(test_pred != y_test)[0]
print("mis_idx[:8]    =", mis_idx[:8])

num_to_show = 8
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, idx in zip(axes.flatten(), mis_idx[:num_to_show]):
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"True: {y_test[idx]}  Pred: {test_pred[idx]}")
    ax.axis('off')

plt.suptitle("Misclassified MNIST Test Samples")
plt.tight_layout()
plt.show()