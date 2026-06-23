import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.tree import plot_tree
# Load the Iris dataset
iris = load_iris()

# Display first 5 flower measurements
print("First 5 flower measurements:")
print(iris.data[:5])

# Display first 5 target labels
print("\nFirst 5 target labels:")
print(iris.target[:5])

# Convert dataset into a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add species column
df['species'] = iris.target

# Convert numerical labels to flower names
df['species'] = df['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

# Display first 5 rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Dataset information
print("\nDataset Information:")
df.info()

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Count of each species
print("\nNumber of flowers in each species:")
print(df['species'].value_counts())

# -------------------------
# EDA (Commented Out)
# -------------------------

plt.figure(figsize=(8, 5))
sns.countplot(x='species', data=df)
plt.title('Number of Flowers in Each Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

# Pair Plot
sns.pairplot(
    df,
    hue='species',
    height=2.5
)

plt.show()

# -------------------------
# Machine Learning Part
# -------------------------

# Features and Target
X = iris.data
y = iris.target

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)

# Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)

print("\nActual Labels:")
print(y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(15,10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title("Decision Tree for Iris Classification")
plt.show()