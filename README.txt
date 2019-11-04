For this assignment, my code uses an Python object Node with attributes
left, right, attribute name,(for pruning) prediction, error, and total.

My code might run a bit slow, with the average range around 4 to 7 minutes, especially with pruning.

There are 3 extra print statements in the outputs that I used for printing the number of nodes,
the 2 baseline errors (training set and test set), so a total of 5 print statements.
If you do NOT need these numbers, please comment them out or ignore them. My first two lines for
accuracies are all you need.

Note: Keyword "vanilla" is for the whole tree, "prune" is for post-pruning tree
In terminal, run
python [path for train file] [path for test file] [vanilla/prune] [% of train data] [if prune, % of validation set]
