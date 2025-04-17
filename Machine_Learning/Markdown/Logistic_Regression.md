# Logistic Regression

**Logistic Regression** is a statistical method for binary classification problems. It is used when the dependent variable (target) is categorical, often with two classes (binary outcomes), such as:

- Yes/No
- True/False
- 1/0 (e.g., whether a customer will buy a product or not)

In Logistic Regression, the output is a probability (between 0 and 1) that a given input point belongs to a certain class. If the probability is greater than a threshold (usually 0.5), it is classified into one class; otherwise, it is classified into the other.

---

## Key Concepts

### 1. Sigmoid Function
The core of Logistic Regression is the **sigmoid function** (also called the logistic function), which maps any real-valued number into the range (0, 1). It is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where:
- $z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$ is the linear combination of input features.
- $\sigma(z)$ is the probability estimate that the input belongs to class 1.

The sigmoid function produces outputs between 0 and 1, which can be interpreted as probabilities.

---

### 2. Decision Boundary
Logistic Regression finds the best-fitting hyperplane (decision boundary) in a multi-dimensional space that separates classes. This is done using the **log odds** of the outcome and optimizing a cost function (usually **cross-entropy loss** or **log-loss**).

The decision boundary is a line (or hyperplane in multiple dimensions) that separates the predicted probabilities into two regions: one corresponding to class 1 and the other to class 0.

---

### 3. Cost Function
The cost function for Logistic Regression is **log loss** (or binary cross-entropy):

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \Big( y_i \log(h_\theta(x_i)) + (1 - y_i) \log(1 - h_\theta(x_i)) \Big)
$$

Where:
- $y_i$ is the true label (0 or 1).
- $h_\theta(x_i)$ is the predicted probability from the sigmoid function.
- $m$ is the number of training samples.

The goal of logistic regression is to minimize this cost function, which ensures that the predicted probabilities are as close as possible to the true labels.

---

## Steps in Logistic Regression

### 1. Hypothesis
The prediction is made based on the logistic sigmoid function:

$$
h_\theta(x) = \sigma(\theta^T x)
$$

Where $\theta$ is the vector of model parameters and $x$ is the vector of input features.

### 2. Training the Model
We use **gradient descent** (or other optimization techniques) to find the parameters $\theta$ that minimize the cost function. The parameters are updated iteratively until the model converges to the optimal values.

### 3. Prediction
Once the model is trained, it outputs a probability $h_\theta(x)$ that the input $x$ belongs to class 1. If the probability is greater than a threshold (usually 0.5), the model classifies the input as belonging to class 1; otherwise, it classifies it as belonging to class 0.

### 4. Evaluation
The model's performance is evaluated using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**. In the case of an imbalanced dataset, **ROC-AUC** (Receiver Operating Characteristic - Area Under Curve) can also be used to assess model performance.

---

## Model Evaluation Metrics

- **Accuracy**: Proportion of correct predictions (both true positives and true negatives) out of all predictions.
- **Precision**: The proportion of true positive predictions out of all predicted positives.
- **Recall (Sensitivity)**: The proportion of true positives out of all actual positives.
- **F1-Score**: The harmonic mean of precision and recall. It is used when the class distribution is imbalanced.
- **Confusion Matrix**: A table that describes the performance of the classification model. It includes:
  - **True Positives (TP)**: The number of positive samples correctly classified.
  - **True Negatives (TN)**: The number of negative samples correctly classified.
  - **False Positives (FP)**: The number of negative samples incorrectly classified as positive.
  - **False Negatives (FN)**: The number of positive samples incorrectly classified as negative.
  
---

## Regularization in Logistic Regression
To prevent overfitting, **regularization** techniques can be applied to Logistic Regression:

- **L1 Regularization (Lasso)**: Adds the absolute value of the coefficients to the cost function. It can drive some coefficients to zero, leading to sparse models.
- **L2 Regularization (Ridge)**: Adds the squared value of the coefficients to the cost function. It penalizes large coefficients and helps to prevent overfitting.

Regularization is controlled by a parameter $\lambda$ (also known as the **regularization strength**), which determines how much regularization is applied. Larger values of $\lambda$ lead to stronger regularization.

---

## Multi-Class Classification
Logistic Regression is inherently a **binary classifier**, but it can be extended to multi-class classification problems using strategies like:

1. **One-vs-Rest (OvR)**: In this approach, a binary classifier is trained for each class, distinguishing that class from all other classes.
2. **Softmax Regression**: This generalization of logistic regression is used for multi-class classification. It computes a probability distribution over multiple classes using the softmax function.

---

## Conclusion
Logistic Regression is a simple yet powerful algorithm for binary classification tasks. It is based on the principles of probability and uses the sigmoid function to model the probability of an event. While it is primarily used for binary classification, with extensions like **one-vs-rest** and **softmax regression**, it can also be applied to multi-class problems. 

Regularization techniques such as L1 and L2 regularization help to prevent overfitting, making logistic regression robust for various real-world applications.
