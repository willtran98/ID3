import pandas as pd
import numpy as np
import math
import sys
import os

node_count = 0


class Tree:
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
        # For pruning
        self.predict = None
        self.total = None
        self.error = None


def preprocess(file_name):
    data = pd.read_csv(file_name, header=None, encoding='utf-8')
    processed_dt = data

    for column in data:
        dum = data[column].str.get_dummies()
        # Delete column
        processed_dt = processed_dt.drop(column, axis=1)
        # Add new binary columns and save new data
        processed_dt = pd.concat([processed_dt, dum], axis=1, sort=False)

    processed_dt = processed_dt.drop([processed_dt.columns[-1]], axis=1)
    # Save new file
    processed_dt.to_csv('new_' + file_name, index=False)


def create_node(data):
    global node_count
    node_count += 1

    # Base case 1: when all outputs are the same
    positive, negative = data[data[' <=50K'] == 1], data[data[' <=50K'] == 0]

    # If outputs are all 1, return leaf with data=1
    if positive[' <=50K'].count() == data[' <=50K'].count():
        leaf = Tree()
        leaf.data = 1
        return leaf

    # If outputs are all 0, return leaf with data=0
    if negative[' <=50K'].count() == data[' <=50K'].count():
        leaf = Tree()
        leaf.data = 0
        return leaf

    # Base case 2: when all inputs are the same
    check = 0
    for col in data:
        sub1, sub2 = data[data[col] == 1], data[data[col] == 0]
        if sub1[col].count() == data[col].count() or sub2[col].count() == data[col].count() or col == ' <=50K':
            continue
        else:
            check = 1
            break

    # Return the leaf with data=majority output
    if check == 0 or data[' <=50K'].count() == 0:
        if positive[' <=50K'].count() >= negative[' <=50K'].count():
            leaf = Tree()
            leaf.data = 1
            return leaf
        else:
            leaf = Tree()
            leaf.data = 0
            return leaf

    # Algorithm to choose root node
    # First, calculate entropy of class label
    f1, f2 = (positive[' <=50K'].count()) / (data[' <=50K'].count()), (negative[' <=50K'].count()) / (data[' <=50K'].count())
    entropy = - f1 * math.log(f1, 2) - f2 * math.log(f2, 2)

    # Second, loop through columns to find root attribute
    max_gain, max_col = -1, ''
    for col in data:
        sub1, sub2 = data[data[col] == 1], data[data[col] == 0]
        if sub1[col].count() == data[col].count() or sub2[col].count() == data[col].count() or col == ' <=50K':
            continue

        # Calculate entropy here
        ssub11, ssub12 = sub1[sub1[' <=50K'] == 1], sub1[sub1[' <=50K'] == 0]
        ssub21, ssub22 = sub2[sub2[' <=50K'] == 1], sub2[sub2[' <=50K'] == 0]

        frac11, frac12 = ssub11[' <=50K'].count() / sub1[' <=50K'].count(), ssub12[' <=50K'].count() / sub1[' <=50K'].count()
        frac21, frac22 = ssub21[' <=50K'].count() / sub2[' <=50K'].count(), ssub22[' <=50K'].count() / sub2[' <=50K'].count()

        if frac11 == 0:
            frac11 = 1
        if frac12 == 0:
            frac12 = 1
        if frac21 == 0:
            frac21 = 1
        if frac22 == 0:
            frac22 = 1
        entropy1 = - frac11 * math.log(frac11, 2) - frac12 * math.log(frac12, 2)
        entropy2 = - frac21 * math.log(frac21, 2) - frac22 * math.log(frac22, 2)

        # Calculate info gain here
        frac1, frac2 = sub1[' <=50K'].count() / data[' <=50K'].count(), sub2[' <=50K'].count() / data[' <=50K'].count()
        info_gain = entropy - (frac1*entropy1 + frac2*entropy2)
        if info_gain > max_gain:
            max_gain = info_gain
            max_col = col

    # create new node with the attribute with maximum info gain
    new_node = Tree()
    new_node.data = max_col
    if positive[' <=50K'].count() >= negative[' <=50K'].count():
        new_node.predict = 1
    else:
        new_node.predict = 0
    # Right tree is 1, left tree 0
    left_node, right_node = data[data[max_col] == 0], data[data[max_col] == 1]

    # RECURSION HERE, GRADERS PAY ATTENTION!!! >_<
    new_node.left = create_node(left_node)
    new_node.right = create_node(right_node)

    return new_node


def accuracy(data, tree):
    # Iterate through the dataframe to count correct predictions
    count = 0
    for i in range(len(data)):
        t = tree
        while t.left is not None or t.right is not None:
            if data.loc[i, t.data] == 0:
                t = t.left
            else:
                t = t.right

        prediction = t.data
        if prediction == data.loc[i, ' <=50K']:
            count += 1

    # Return the accuracy of predictions
    return count/data[' <=50K'].count()


def prune(v_data, tree, start):
    # Iterate through the dataframe to count incorrect predictions for each node
    for i in range(start, len(v_data)):
        t = tree
        while t.left is not None or t.right is not None:
            if not t.predict == v_data.loc[i, ' <=50K']:
                t.error += 1
                t.total += 1
            else:
                t.total += 1

            if v_data.loc[i, t.data] == 0:
                t = t.left
            else:
                t = t.right

        prediction = t.data
        if prediction == v_data.loc[i, ' <=50K']:
            t.total += 1
        else:
            t.total += 1
            t.error += 1

    # pruning
    while 0 != 1:
        t = tree
        check = reduce_error(t)
        # If no further pruning is done, break out of the loop
        if check == 0:
            break

    # Return the pruned tree
    return tree


def reduce_error(tree):
    t = tree
    global node_count
    if t.left is None and t.right is None:
        return 0

    # Actual pruning
    if (t.left.left is None and t.left.right is None) or (t.right.left is None and t.right.right is None):
        if t.error is None or t.total is None or t.left.error is None or t.left.total is None or t.right.error is None or t.right.total is None:
            return 0
        if t.error / t.total < t.left.error / t.left.total and t.error / t.total < t.right.error / t.right.total:
            t.left = None
            t.right = None
            node_count -= 1
            return 1

    return reduce_error(t.right) + reduce_error(t.left)


def main():
    # Building Phase
    preprocess(sys.argv[1])
    preprocess(sys.argv[2])
    train_name = 'new_' + sys.argv[1]
    test_name = 'new_' + sys.argv[2]
    train_dt = pd.read_csv(train_name, encoding='utf-8')
    test_dt = pd.read_csv(test_name, encoding='utf-8')
    train_dt2 = train_dt.head(int(len(train_dt) * (int(sys.argv[4]) / 100)))

    # Operational Phase
    train_tree = create_node(train_dt2)
    # Pruning
    if sys.argv[3] == 'prune':
        validation_dt = train_dt.tail(int(len(train_dt) * (int(sys.argv[5]) / 100)))
        train_tree = prune(validation_dt, train_tree, len(train_dt) - int(len(train_dt) * (int(sys.argv[5]) / 100)))

    # Prediction
    a_train = accuracy(train_dt2, train_tree)
    a_test = accuracy(test_dt, train_tree)
    print("Train set accuracy: ", a_train)
    print("Test set accuracy: ", a_test)

    # Baseline default error
    positive, negative = train_dt[train_dt[' <=50K'] == 1], train_dt[train_dt[' <=50K'] == 0]
    if positive[' <=50K'].count() >= negative[' <=50K'].count():
        p1 = train_dt2[train_dt2[' <=50K'] == 1]
        print('Train baseline default error: ', p1[' <=50K'].count() / train_dt2[' <=50K'].count())
        p2 = test_dt[test_dt[' <=50K'] == 1]
        print('Test baseline default error: ', p2[' <=50K'].count() / test_dt[' <=50K'].count())
    else:
        p1 = train_dt2[train_dt2[' <=50K'] == 0]
        print('Train baseline default error: ', p1[' <=50K'].count() / train_dt2[' <=50K'].count())
        p2 = test_dt[test_dt[' <=50K'] == 0]
        print('Test baseline default error: ', p2[' <=50K'].count() / test_dt[' <=50K'].count())

    global node_count
    print('Number of nodes: ', node_count)


# Calling main function
if __name__ == "__main__":
    main()


