# igaltal-Machine-Learning - Decision Tree Project

Project Description
-------------------
This project involves building a Decision Tree classifier from scratch. The classifier can use different impurity measures (Gini and Entropy) and supports pruning techniques such as depth pruning and chi-square pruning. The project also includes functions for evaluating the accuracy of the decision tree on training and validation datasets.

Project Structure
-----------------
- chi_table: A dictionary containing chi-square critical values for different degrees of freedom and p-values.
- calc_gini(data): Function to calculate the Gini impurity of a dataset.
- calc_entropy(data): Function to calculate the entropy of a dataset.
- DecisionNode: A class representing a node in the decision tree.
    - __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False): Initializes a decision node.
    - calc_node_pred(self): Calculates the prediction of the node.
    - add_child(self, node, val): Adds a child node.
    - calc_feature_importance(self, n_total_sample): Calculates the feature importance.
    - goodness_of_split(self, feature): Calculates the goodness of split for a given feature.
    - split(self): Splits the current node based on the impurity function.
- DecisionTree: A class representing the decision tree.
    - __init__ Initializes the decision tree.
    - build_tree(self): Builds the decision tree.
    - predict(self, instance): Predicts the class label for a given instance.
    - calc_accuracy(self, dataset): Calculates the accuracy of the decision tree on a given dataset.
    - depth(self): Returns the depth of the tree.
- depth_pruning(X_train, X_validation): Function to calculate training and validation accuracies for different depths using the best impurity function and gain ratio.
- chi_pruning(X_train, X_test): Function to calculate training and validation accuracies for different chi values using the best impurity function and gain ratio.
- count_nodes(node): Function to count the number of nodes in the decision tree.

Dependencies
------------
- numpy
- matplotlib

Usage
-----
1. Import the necessary functions and classes from the project.
2. Load your dataset into a numpy array.
3. Create an instance of the DecisionTree class with the desired impurity function, chi value, maximum depth, and gain ratio flag.
4. Build the tree using the build_tree method.
5. Predict the class label for new instances using the predict method.
6. Evaluate the accuracy of the decision tree using the calc_accuracy method.
7. Use depth_pruning and chi_pruning functions to experiment with different pruning techniques.

Example
-------
```
import numpy as np
from decision_tree import DecisionTree, calc_gini, calc_entropy, depth_pruning, chi_pruning

# Load your dataset
data = np.array([...])  # Replace with your dataset

# Create a decision tree instance
tree = DecisionTree(data=data, impurity_func=calc_entropy, max_depth=10, gain_ratio=True)

# Build the tree
tree.build_tree()

# Predict class labels for new instances
prediction = tree.predict(new_instance)

# Evaluate accuracy
accuracy = tree.calc_accuracy(validation_data)

# Perform depth pruning
train_acc, val_acc = depth_pruning(X_train, X_validation)

# Perform chi-square pruning
chi_train_acc, chi_val_acc, depths = chi_pruning(X_train, X_test)
```
