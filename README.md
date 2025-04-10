### Binary and Multi-class Classification in PyTorch

This repository contains two Python scripts that demonstrate simple **binary classification** and **multi-class classification** using PyTorch. The first script implements a basic binary classification model (a form of logistic regression using a small neural network) on a synthetic dataset. The second script demonstrates a multi-class classification model on a dataset with multiple classes (similar concept to the Iris dataset, but here using a generated dataset with four clusters). These examples are designed to be beginner-friendly, with clear code and comments to help you understand how data is processed, models are built, and training is performed step by step.

In the binary classification example (`model_1.py`), we generate a two-class dataset of points in a circle pattern and train a simple neural network with a sigmoid output to distinguish the classes. In the multi-class example (`multiclass_model.py`), we generate a dataset of points belonging to four clusters and train a neural network with a softmax output to classify each point into one of four classes. Both scripts illustrate core machine learning tasks like data splitting, model definition, training loops, loss computation, and accuracy evaluation using PyTorch and scikit-learn.
## Project Files

    `model_1.py` – Binary Classification Model: Creates a synthetic two-class dataset (two interlocking circles) and trains a simple neural network (logistic regression model with hidden layers) to classify the points into two classes (class 0 vs 1).

    `multiclass_model.py` – Multi-class Classification Model: Generates a synthetic dataset of points in four clusters (you can think of this as a stand-in for a dataset like Iris with multiple classes) and trains a neural network to classify points into one of four categories.

Below, we provide detailed explanations and code snippets for each script.
## `model_1.py`: **Binary Classification Model**

This script builds and trains a binary classifier using PyTorch on a toy dataset. Key steps in model_1.py:

   - **Data Generation**: Uses sklearn.datasets.make_circles to create a synthetic dataset of 1000 points in two classes (forming concentric circles).

   - **Data Preparation**: Converts the NumPy arrays to PyTorch tensors and splits the data into training and test sets (80% train, 20% test).

   - **Model Definition**: Defines CircleModelVO, a simple neural network with two hidden layers (10 units each, using ReLU activation) and an output layer with 1 unit for binary classification (predicting 0 or 1).

   - **Training Loop**: Uses Binary Cross-Entropy Loss (BCELoss) for the binary classification and Stochastic Gradient Descent (SGD) optimizer. Trains the model for 2000 epochs, and computes accuracy using a custom accuracy_fn.

   - **Output**: Prints the training and testing accuracy and loss every 100 epochs to show progress.

<details> <summary><strong>model_1.py</strong> – <em>Binary Classification Model (expand to view code)</em></summary>

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
import torch
from torch import nn
# from pathlib import Path  # (This import is unused in the code)

# 1. Generate synthetic binary classification data (two circles)
circle_points = 1000
X, Y = make_circles(n_samples=circle_points, noise=0.03, random_state=42)

# 2. Convert NumPy arrays to PyTorch tensors (required for PyTorch computations)
X = torch.from_numpy(X).type(torch.float32)    # features tensor of shape [1000, 2]
Y = torch.from_numpy(Y).type(torch.float32)    # labels tensor of shape [1000], values are 0 or 1

# 3. Define a simple neural network model for binary classification
class CircleModelVO(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear layer transformations
        self.layer_1 = nn.Linear(in_features=2, out_features=10)   # hidden layer 1
        self.layer_2 = nn.Linear(in_features=10, out_features=10)  # hidden layer 2
        self.layer_3 = nn.Linear(in_features=10, out_features=1)   # output layer (1 output for binary classification)
        self.relu = nn.ReLU()  # ReLU activation function for non-linearity

    def forward(self, x):
        # Forward pass: input x goes through layer 1 -> ReLU -> layer 2 -> ReLU -> layer 3
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.layer_3(x)
        return x
```

# 4. Split the data into training and testing sets
``` python 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

# 5. Initialize the model, loss function, and optimizer
```python
model_1 = CircleModelVO()                          # instantiate the model
loss_function = nn.BCELoss()                       # binary cross-entropy loss for binary classification
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)  # stochastic gradient descent optimizer with learning rate 0.1
```
# 6. Define a helper function to calculate accuracy

```python
def accuracy_fn(y_true, y_pred):
    """
    Computes the percentage of correct predictions.
    y_true and y_pred are tensors of the same shape.
    """
    correct = torch.eq(y_true, y_pred).sum().item()  # count how many predictions match the true labels
    total = len(y_true)
    return (correct / total) * 100  # percentage of correct predictions
```
# 7. Set manual seed for reproducibility (ensures same random initialization every run)
```python 
torch.manual_seed(42)
```

# 8. Train the model for a certain number of epochs
```python
epochs = 2000
for epoch in range(epochs):
    # Set model to training mode
    model_1.train()

    # Forward pass on training data
    y_pred_logits = model_1(X_train).squeeze()      # raw model outputs (logits) for each training sample
    y_pred_probs = torch.sigmoid(y_pred_logits)     # apply sigmoid to get probabilities between 0 and 1
    y_pred_labels = torch.round(y_pred_probs)       # convert probabilities to 0 or 1 predictions (threshold at 0.5)

    # Compute training loss (BCELoss expects probabilities as input for binary classification)
    loss = loss_function(y_pred_probs, y_train)
    # Compute training accuracy using the rounded predictions
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred_labels)

    # Backpropagation preparation: zero out previous gradients
    optimizer.zero_grad()
    # Compute gradients for each parameter via backpropagation
    loss.backward()
    # Update parameters using the optimizer (gradient descent step)
    optimizer.step()

    # Evaluation on test data (inference mode disables gradient computation)
    model_1.eval()
    with torch.inference_mode():
        # Forward pass on test data
        test_logits = model_1(X_test).squeeze()
        test_probs = torch.sigmoid(test_logits)               # get probabilities for test data
        test_preds = torch.round(test_probs)                  # get binary predictions for test data
        # Calculate test loss and accuracy
        test_loss = loss_function(test_probs, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds)

    # Print metrics every 100 epochs for monitoring
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | "
              f"Train Loss: {loss:.4f}, Train Acc: {acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
```

</details>

## `multiclass_model.py`: **Multi-class Classification Model**

This script builds and trains a classifier for multiple classes using PyTorch. Key steps in multiclass_model.py:

   - Data Generation: Uses sklearn.datasets.make_blobs to generate 1000 sample points with 2 features, grouped into 4 distinct clusters/classes.

   - Data Preparation: Converts the generated data to PyTorch tensors and splits into training and test sets (80% train, 20% test). There is also a helper function showData() that can plot the data points with their cluster colors (useful for visualization if running in an interactive environment).

   - Model Definition: Defines MultiDef, a neural network class with two hidden layers (8 units each) and an output layer of size 4 (for the four classes). ReLU is used for hidden layer activation. (An alternative definition using nn.Sequential is shown in comments for reference.)

    - Training Loop: Uses Cross Entropy Loss (CrossEntropyLoss) for multi-class classification and SGD optimizer. Trains for 120 epochs, computing the predicted class by applying softmax and selecting the class with highest probability.

    **Output**: Prints the training and testing loss and accuracy every 10 epochs to track performance.

<details> <summary><strong>multiclass_model.py</strong> – <em>Multi-class Classification Model (expand to view code)</em></summary>

```python 
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
```
# 1. Generate synthetic multi-class data (4 clusters/classes)
```python
RANDOM_STATE = 42
X, y = make_blobs(n_samples=1000, n_features=2, centers=4, cluster_std=1.0, random_state=RANDOM_STATE)
```
# 2. Convert data to PyTorch tensors

```python
X_blob = torch.from_numpy(X).type(torch.float32)  # features tensor [1000, 2]
y_blob = torch.from_numpy(y).type(torch.long)     # labels tensor [1000], values 0,1,2,3 for each class
```
# 3. (Optional) Define a helper function to visualize the dataset

```python
def showData():
    """
    Plot the 2D data points, coloring them by their class.
    """
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.show()
```
# 4. Define a neural network for multi-class classification
```python
class MultiDef(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        # Define layers: two hidden layers and one output layer
        self.layer1 = nn.Linear(in_features=input_features, out_features=hidden_units)
        self.layer2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.layer3 = nn.Linear(in_features=hidden_units, out_features=output_features)
        # Note: ReLU activation will be applied in forward pass (or could be part of nn.Sequential)

    def forward(self, x):
        # Forward pass through the layers with ReLU activations for hidden layers
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = self.layer2(x)
        x = nn.ReLU()(x)
        x = self.layer3(x)  # outputs raw scores (logits) for each class
        return x

# Alternate model definition (not used, but shown for reference):
# model = nn.Sequential(
#     nn.Linear(input_features, hidden_units),
#     nn.ReLU(),
#     nn.Linear(hidden_units, hidden_units),
#     nn.ReLU(),
#     nn.Linear(hidden_units, output_features)
# )
```
# 5. Initialize the model
```python 
model = MultiDef(input_features=2, output_features=4, hidden_units=8)
```

# 6. Split data into training and test sets
```python 
X_train, X_test, y_train, y_test = train_test_split(X_blob, y_blob, test_size=0.2, random_state=RANDOM_STATE)
```

# 7. Set up loss function and optimizer for training
```python
loss_function = nn.CrossEntropyLoss()                      # suitable for multi-class classification
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)   # SGD optimizer with a small learning rate
```
# 8. Training loop for a fixed number of epochs
````python
epochs = 120
for epoch in range(epochs):
    model.train()  # set model to training mode

    # Forward pass on training data to get raw predictions (logits)
    y_logits = model(X_train)
    # Convert logits to predicted class labels:
    #   torch.softmax -> probabilities, then argmax to get index of highest prob (predicted class)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    # Compute training loss (CrossEntropyLoss expects raw logits and true class indices)
    train_loss = loss_function(y_logits, y_train)

    # Backpropagation
    optimizer.zero_grad()   # zero out previous gradients
    train_loss.backward()   # compute gradients for this batch
    optimizer.step()        # update model parameters

    # Calculate training accuracy for this epoch
    train_acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # Evaluate on test data (no gradient needed)
    model.eval()
    with torch.inference_mode():
        # Forward pass on test set
        test_logits = model(X_test)
        # Compute test loss
        test_loss = loss_function(test_logits, y_test)
        # Get predicted classes for test set
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        # Calculate test accuracy
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # Print metrics every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f}  | Test Accuracy: {test_acc:.2f}%")
        print("-" * 50)

# Note: accuracy_fn is assumed to be defined similarly as in model_1.py
def accuracy_fn(y_true, y_pred):
    """
    Utility function to calculate accuracy (% of correct predictions).
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_true)) * 100
```

</details>
## Installation

To run these examples, you will need Python 3.x installed. We recommend using a virtual environment to avoid conflicts with other packages. Follow the steps below to set up the project:

    Clone the repository (or download the ZIP):

```
git clone https://github.com/DahalAb1/multiclass-classifier-pytorch.git
```

Install the required packages. You can install the dependencies using `pip`:
```
    pip install torch torchvision scikit-learn matplotlib

    Note: The above command installs PyTorch (CPU version) along with scikit-learn and matplotlib. If you have a CUDA-compatible GPU and want to use it, consider installing the appropriate PyTorch version from the official PyTorch Website.
```
## Usage

After installing the dependencies, you can run each script directly to train the models and see the output:

   ` Binary Classification Demo: Run model_1.py to train the binary classifier.`

`python model_1.py`

This will generate the synthetic circle data and train the neural network. The program will print the training and testing accuracy/loss every 100 epochs. After 2000 epochs, you should see the final accuracy printed in the console.

Multi-class Classification Demo: Run multiclass_model.py to train the multi-class classifier.

    python multiclass_model.py

    This will generate a 4-class dataset and train the network for 120 epochs, printing the training and testing metrics every 10 epochs. You can observe the accuracy improving over epochs. (Optional: If you want to visualize the dataset clusters, you can call the showData() function in the script or run those lines in a Jupyter notebook to see a scatter plot of the generated data.)*

Both scripts run purely in the console and will output training progress there. There is no file output by default. You can modify the scripts to save models or plots as needed.
Future Plans

** Some ideas for future improvements and extensions of this project include:**

    - Use Real Datasets: Replace the synthetic data with real-world datasets (e.g., using the Iris dataset for multi-class classification or another binary classification dataset) to further validate the models.

    - Model Enhancements: Experiment with deeper networks or different architectures (adding more layers or using different activation functions) to see how they affect performance. Also, try different optimizers (Adam, RMSprop) or tuning learning rates for potentially faster convergence.

    - Evaluation & Visualization: Add more evaluation metrics (such as precision, recall, confusion matrix) and visualization of decision boundaries or loss curves to better understand the model performance.

    - Packaging and Tests: Refactor the code into functions or classes for easier reuse, and add unit tests. Possibly create a Jupyter notebook for an interactive tutorial format, and provide a requirements file or setup script for easy installation.