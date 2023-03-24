import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso

def generate_data(n_samples, slope=3, intercept=2, noise=1):
    """Generate random data for linear regression."""
    np.random.seed(42)
    X = np.random.rand(n_samples, 1)
    y = intercept + slope * X + noise * np.random.randn(n_samples, 1)
    return X, y

def train_and_evaluate_model(model, X, y):
    """Train and evaluate a linear regression model."""
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    return np.mean(scores)

def main():
    """Main function for running the script."""
    X, y = generate_data(n_samples=1000, slope=5, intercept=2, noise=2)
    plt.scatter(X, y)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso()
    }

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f"{name}: mean R-squared score = {np.mean(scores):.4f} (std = {np.std(scores):.4f})")

    param_grid = {'alpha': np.logspace(-3, 3, 7)}
    ridge_model = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
    ridge_model.fit(X, y)
    print(f"Best Ridge Regression model: alpha = {ridge_model.best_params_['alpha']:.4f}, R-squared score = {ridge_model.best_score_:.4f}")

if __name__ == '__main__':
    main()
