# Linear Regression

Linear regression is a statistical method used to model the relationship between a dependent (response) variable and one or more independent (predictor) variables. It assumes that there is a linear relationship between the variables, making it both simple and interpretable.

## Key Concepts:

### 1. The Linear Model:

The basic form of the linear regression equation is:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon
$$

Where:

- **Y**: The dependent variable (what you're trying to predict).
- **X₁, X₂, ..., Xₙ**: The independent variables (predictors).
- **β₀**: The intercept (value of Y when all X's are 0).
- **β₁, β₂, ..., βₙ**: The coefficients (slopes) for each of the independent variables.
- **ε**: The error term (residuals) that accounts for the difference between the observed and predicted values.

### 2. Objective:

The goal is to find the best-fitting line (or hyperplane in the case of multiple predictors) that minimizes the difference between the predicted values and the actual values. This is achieved by minimizing the **sum of squared residuals** (the least squares criterion).

### 3. Assumptions of Linear Regression:

- **Linearity**: The relationship between the independent and dependent variables is linear.
- **Independence**: Observations are independent of each other.
- **Homoscedasticity**: The variance of errors is constant across all values of the independent variables.
- **Normality**: The errors are normally distributed (important for hypothesis testing).

## Types of Linear Regression:

### 1. Simple Linear Regression:

Involves one independent variable. The model is represented as:

$$
Y = \beta_0 + \beta_1 X + \epsilon
$$

This results in a straight line when plotted.

### 2. Multiple Linear Regression:

Involves two or more independent variables. The model is extended to:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon
$$

This results in a hyperplane in a multidimensional space.

## How It Works:

### 1. Finding the Best Fit:

The model finds the values of **β₀, β₁, ..., βₙ** that minimize the **residual sum of squares (RSS)**:

$$
RSS = \sum (Y_{\text{observed}} - Y_{\text{predicted}})^2
$$

Where **$Y_{\text{observed}}$** are the observed values and **$Y_{\text{predicted}}$** are the predicted values. This is typically solved using **Ordinary Least Squares (OLS)** or other optimization techniques.

### 2. Evaluating the Model:

Once the model is trained, its performance is evaluated using metrics like:

- **R-squared**: Measures how well the model explains the variance in the dependent variable.
- **Adjusted R-squared**: Adjusted for the number of predictors.
- **Mean Squared Error (MSE)** or **Root Mean Squared Error (RMSE)**: Measures the average squared difference between actual and predicted values.

### 3. Interpretation of Coefficients:

- The coefficient **$\beta_1$** represents the change in **$Y$** for a one-unit change in **$X_1$**, assuming all other predictors are constant.
- If **$\beta_1 = 2$**, then for each unit increase in **$X_1$**, **$Y$** increases by 2 units.

## Practical Use Cases:

- Predicting house prices based on features like square footage, number of bedrooms, etc.
- Predicting stock prices using historical data.
- Analyzing the relationship between education level and income.