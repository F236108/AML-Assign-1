"""
Task 4 – Linear Regression from Scratch using NumPy only.

Contains:
  - hypothesis(X, theta)
  - compute_cost(X, y, theta)
  - fit_normal(X, y)
  - fit_batch_gd(X, y, alpha, n_iters)
  - fit_sgd(X, y, alpha, n_epochs, seed)
  - rmse(y_true, y_pred)
"""
import numpy as np

def hypothesis(X, theta):
    """
    Computes predictions h_theta(X) = X @ theta.
    X: shape (m, d+1)
    theta: shape (d+1,) or (d+1, 1)
    """
    return np.dot(X, theta)

def compute_cost(X, y, theta):
    """
    Computes mean squared error cost:
    J(theta) = (1 / (2 * m)) * sum((X @ theta - y)^2)
    """
    m = len(y)
    preds = hypothesis(X, theta)
    errors = preds - y
    cost = (1.0 / (2.0 * m)) * np.sum(errors ** 2)
    return cost

def fit_normal(X, y):
    """
    Computes closed-form solution using Normal Equation:
    theta = (X^T @ X)^(-1) @ X^T @ y
    """
    # Using np.linalg.pinv for numerical stability or np.linalg.inv
    XtX = np.dot(X.T, X)
    XtY = np.dot(X.T, y)
    theta = np.linalg.solve(XtX, XtY)
    return theta

def fit_batch_gd(X, y, alpha, n_iters):
    """
    Batch Gradient Descent:
    theta := theta - alpha * (1/m) * X^T @ (X @ theta - y)
    Returns: (theta, cost_history)
    """
    m, d = X.shape
    theta = np.zeros(d)
    cost_history = [compute_cost(X, y, theta)]
    
    for _ in range(n_iters):
        preds = hypothesis(X, theta)
        errors = preds - y
        grad = (1.0 / m) * np.dot(X.T, errors)
        theta = theta - alpha * grad
        cost_history.append(compute_cost(X, y, theta))
        
    return theta, cost_history

def fit_sgd(X, y, alpha, n_epochs, seed=42):
    """
    Stochastic Gradient Descent (SGD):
    Updates theta sample-by-sample for each epoch.
    Returns: (theta, cost_history)
    """
    m, d = X.shape
    theta = np.zeros(d)
    cost_history = [compute_cost(X, y, theta)]
    
    np.random.seed(seed)
    indices = np.arange(m)
    
    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        for i in indices:
            xi = X[i:i+1] # shape (1, d)
            yi = y[i:i+1] # shape (1,)
            pred_i = np.dot(xi, theta)
            err_i = pred_i - yi
            grad_i = xi.T.flatten() * err_i[0]
            theta = theta - alpha * grad_i
            
        cost_history.append(compute_cost(X, y, theta))
        
    return theta, cost_history

def rmse(y_true, y_pred):
    """
    Computes Root Mean Squared Error:
    RMSE = sqrt( (1 / m) * sum((y_true - y_pred)^2) )
    """
    m = len(y_true)
    return np.sqrt( (1.0 / m) * np.sum((y_true - y_pred) ** 2) )
