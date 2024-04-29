import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
                 0.25: 1.32,
                 0.1: 2.71,
                 0.05: 3.84,
                 0.0001: 100000},
             2: {0.5: 1.39,
                 0.25: 2.77,
                 0.1: 4.60,
                 0.05: 5.99,
                 0.0001: 100000},
             3: {0.5: 2.37,
                 0.25: 4.11,
                 0.1: 6.25,
                 0.05: 7.82,
                 0.0001: 100000},
             4: {0.5: 3.36,
                 0.25: 5.38,
                 0.1: 7.78,
                 0.05: 9.49,
                 0.0001: 100000},
             5: {0.5: 4.35,
                 0.25: 6.63,
                 0.1: 9.24,
                 0.05: 11.07,
                 0.0001: 100000},
             6: {0.5: 5.35,
                 0.25: 7.84,
                 0.1: 10.64,
                 0.05: 12.59,
                 0.0001: 100000},
             7: {0.5: 6.35,
                 0.25: 9.04,
                 0.1: 12.01,
                 0.05: 14.07,
                 0.0001: 100000},
             8: {0.5: 7.34,
                 0.25: 10.22,
                 0.1: 13.36,
                 0.05: 15.51,
                 0.0001: 100000},
             9: {0.5: 8.34,
                 0.25: 11.39,
                 0.1: 14.68,
                 0.05: 16.92,
                 0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    if data.size == 0:
        return 0  # Handling the case where the dataset is empty

    total_instances = len(data)
    labels = np.unique(data[:, -1])  # Using -1 directly to access the last column

    for label in labels:
        count = len(data[data[:, -1] == label])
        prob = count / total_instances
        gini += prob ** 2

    return 1 - gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    if data.size == 0:
        return 0

    total_instances = len(data)
    # Last column contains the class labels
    labels = np.unique(data[:, -1])
    entropy = 0.0

    for label in labels:
        count = len(data[data[:, -1] == label])
        probability = count / total_instances
        # Only add the term if probability is not zero
        if probability > 0:
            entropy += probability * np.log2(probability)

    return -entropy


class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = len(
            np.unique(self.data[:, -1])) == 1 or self.depth >= max_depth  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        if len(self.data) == 0:
            return None
        labels, counts = np.unique(self.data[:, -1], return_counts=True)
        return labels[np.argmax(counts)]

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.

        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in
        self.feature_importance
        """
        if self.terminal or self.feature == -1:
            return

        # Calculate the decrease in impurity this split generated
        impurity_before_split = self.impurity_func(self.data)
        impurity_after_split = 0
        for child in self.children:
            weight = len(child.data) / n_total_sample
            impurity_after_split += weight * self.impurity_func(child.data)

        # Calculate the information gain
        information_gain = impurity_before_split - impurity_after_split
        self.feature_importance = information_gain / impurity_before_split if impurity_before_split != 0 else 0

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting
                  according to the feature values.
        """
        groups = {}
        for value in np.unique(self.data[:, feature]):
            subset = self.data[self.data[:, feature] == value]
            groups[value] = subset

        impurity_before_split = self.impurity_func(self.data)
        total_instances = len(self.data)
        impurity_after_split = sum(
            (len(subset) / total_instances) * self.impurity_func(subset) for subset in groups.values())
        gain = impurity_before_split - impurity_after_split

        if self.gain_ratio and gain > 0:
            split_info = -sum(
                (len(subset) / total_instances) * np.log2(len(subset) / total_instances) for subset in groups.values()
                if len(subset) > 0)
            return gain / split_info if split_info != 0 else gain, groups
        return gain, groups

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        if self.terminal:
            return  # Stop splitting if the node is terminal

        best_gain = -float('inf')
        best_feature = None
        best_split = None

        for feature in range(self.data.shape[1] - 1):
            gain, groups = self.goodness_of_split(feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_split = groups

        if best_feature is None:
            self.terminal = True
            return

        self.feature = best_feature
        for value, subset in best_split.items():
            if len(subset) > 0:
                child_node = DecisionNode(
                    data=subset,
                    impurity_func=self.impurity_func,
                    depth=self.depth + 1,
                    chi=self.chi,
                    max_depth=self.max_depth,
                    gain_ratio=self.gain_ratio
                )
                self.children.append(child_node)
                self.children_values.append(value)
                child_node.split()


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the tree
        self.impurity_func = impurity_func  # the impurity function to be used in the tree
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio  #
        self.root = None  # the root node of the tree

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset.
        You are required to fully grow the tree until all leaves are pure
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = DecisionNode(self.data, self.impurity_func, chi=self.chi, max_depth=self.max_depth,
                                 gain_ratio=self.gain_ratio)
        if not self.root.terminal:
            self.root.split()

    def predict(self, instance):
        """
        Predict a given instance

        Input:
        - instance: an row vector from the dataset. Note that the last element
                    of this vector is the label of the instance.

        Output: the prediction of the instance.
        """
        current_node = self.root
        while not current_node.terminal:
            value = instance[current_node.feature]
            found_child = False
            for i, child in enumerate(current_node.children):
                if current_node.children_values[i] == value:
                    current_node = child
                    found_child = True
                    break
            if not found_child:
                break
        return current_node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset

        Input:
        - dataset: the dataset on which the accuracy is evaluated

        Output: the accuracy of the decision tree on the given dataset (%).
        """
        correct = 0
        # Going over all the instances in the dataset.
        for instance in dataset:
            # Taking the prediction of the instance according to predict function and the actual value.
            pred = self.predict(instance)
            pred_check = instance[-1]
            # Checking if what we predict to the instance is the actual value.
            if pred == pred_check:
                correct += 1

        # Calculate the accuracy of our tree.
        accuracy = (correct / len(dataset)) * 100
        return accuracy

    def depth(self):
        return self.root.depth()


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy
    as a function of the max_depth.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output: the training and validation accuracies per max depth
    """
    training_acc = []
    validation_acc = []
    for max_depth in range(1, 11):
        tree = DecisionTree(data=X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=True)
        tree.build_tree()
        train_acc = tree.calc_accuracy(X_train)
        valid_acc = tree.calc_accuracy(X_validation)
        training_acc.append(train_acc)
        validation_acc.append(valid_acc)

    return training_acc, validation_acc


def chi_pruning(X_train, X_test):
    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc = []
    depths = []
    p_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]

    # Going over all the values of the p_values.
    for val in p_values:
        # Creating a tree that his chi value is according to the p_values and then we calculate the accuracy and the depth of the tree.
        tree = DecisionTree(data=X_train, impurity_func=calc_entropy, chi=val)
        tree.build_tree()
        accuracy_train = tree.calc_accuracy(X_train)
        accuracy_test = tree.calc_accuracy(X_test)
        chi_training_acc.append(accuracy_train)
        chi_testing_acc.append(accuracy_test)
        tree_depth = tree.depth
        depths.append(tree_depth)

    return chi_training_acc, chi_testing_acc, depths


def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of node in the tree.
    """
    n_nodes = 0
    # Checking if the node exist and then count him and count his children nodes.
    if node is not None:
        n_nodes = 1 + sum(count_nodes(child) for child in node.children)
    return n_nodes
