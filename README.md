# Python scikit-learn
 implementing a simple machine learning model using scikit-learn in Python

## 1: Import Libraries
First, let's import the necessary libraries:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```
## Step 2: Prepare the Data
Next, let's generate some random data to use for our model:

```python
# Generate random data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)
```
We're generating 100 random data points with one feature (X) and one label (y). We're using a linear relationship between X and y, with some random noise added in.

## Step 3: Split the Data
Now, let's split our data into training and testing sets:

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
We're using the train_test_split function from scikit-learn to split our data into training and testing sets. We're using a test size of 20%, which means that 20% of the data will be used for testing and the remaining 80% will be used for training.

## Step 4: Train the Model
Next, let's train our linear regression model on the training data:

```python
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
```
We're using scikit-learn's LinearRegression class to create a linear regression model, and then we're calling the fit method to train the model on our training data.

## Step 5: Evaluate the Model
Finally, let's evaluate our model on the testing data:

```python
# Evaluate the model
score = model.score(X_test, y_test)
print("R-squared score:", score)
```
Here, we're using the score method to calculate the R-squared score for our model on the testing data.
And done!
