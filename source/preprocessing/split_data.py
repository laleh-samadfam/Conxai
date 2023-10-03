import numpy as np
from sklearn.model_selection import train_test_split

# Assume X contains features and y contains labels

# Generate sample data (replace with your actual data)
X = np.random.random((100, 10))  # Example: 100 samples, 10 features
y = np.random.randint(0, 3, 100)  # Example: 3 classes

# Split the data into training, validation, and test sets with stratified sampling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Print the class distribution in each set
print("Class distribution in the original dataset:")
print(np.bincount(y) / len(y))

print("Class distribution in the training set:")
print(np.bincount(y_train) / len(y_train))

print("Class distribution in the validation set:")
print(np.bincount(y_val) / len(y_val))

print("Class distribution in the test set:")
print(np.bincount(y_test) / len(y_test))
