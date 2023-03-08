import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def generate_data(n_samples):
    """Generate random data for linear regression."""
    np.random.seed(42)
    X = np.random.rand(n_samples, 1)
    y = 2 + 3 * X + np.random.randn(n_samples, 1)
    return X, y

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Train and evaluate a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    r_squared = model.score(X_test, y_test)
    return r_squared

def main():
    """Main function for running the script."""
    X, y = generate_data(n_samples=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    r_squared = train_and_evaluate_model(X_train, y_train, X_test, y_test)
    print(f"R-squared score: {r_squared:.4f}")

if __name__ == '__main__':
    main()
