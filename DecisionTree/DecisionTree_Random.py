import numpy as np
import sys
import random
from collections import deque
from random import choice, sample
from math import ceil


class ID3Algorithm:


    def __init__(self, data, skip_id3):
        self.attrs = set(list(data.dtype.names)[:-1])
        self.root = TreeNode(data, self.attrs, name='root')
        self.leafs = 0
        self.build(skip_id3)
        self.n, self.leafNodes = self.labelNodes()
        self.depthofleafs = 0

    def __repr__(self):
        '''
        Returns formatted object
        for printing
        '''
        return 'ID3 decision tree: {} nodes & {} leaves'.format(self.getNodesCount(), self.getLeavesCount())
        

    def getNodesCount(self):
        '''
        Returns number of nodes in decision tree
        '''
        return self.n

    def getDepthofLeaves(self):
        '''
        Returns depth of leaves

        '''
        return self.depthofleafs

    def getLeavesCount(self):
        '''
        Returns number of
        leaf nodes present
        '''
        return self.leafs


    def findAccuracy(self, data):
        '''
        Checks accuracy of tree on data examples
        by passing each one through the decision
        tree
        '''
        correct = 0
        for instance in data:
            out = instance['Class']
            curr = self.root
            # iterate to leaf node
            while curr.output == None:
                # get attribute of node split
                if curr.left:
                    attr = curr.left.split
                else:
                    attr = curr.right.split
                if instance[attr]:
                    curr = curr.right
                else:
                    curr = curr.left
            if out == curr.output:
                correct += 1
        #return correct / len(data)
        return (correct / len(data))*100


    def build(self, skip_id3):
        '''
        Builds the decision tree classifier
        from the root node of the tree
        based on the ID3 or random selection
        '''
        if skip_id3:
            self.buildDtree_Random_attr(self.root)
        else:
            self.buildDTree_ID3(self.root)
        

    def buildDTree_ID3(self, node):
        '''
        Helpers method for building
        the decision tree classifier based on ID3 algorithm
        '''
        if len(node.examples) == 0:
            # node doesn't have any examples
            node.output = choice([0, 1])
            self.leafs += 1
            return 

        # handle leaf node cases
        if not node.attrs:
            # available attributes to split on
            # have been exhausted
            val, counts = np.unique(node.examples['Class'], return_counts=True)
            node.output = max(list(zip(counts, val)))[1]
            self.leafs += 1
            return

        if node.isSameClass():
            # node contains examples all with 
            # the same class label
            node.output = node.examples['Class'][0]
            self.leafs += 1
            return

        _, split_attr = findSplitAttribute(node)
        left, right = split(node, split_attr, node.attrs-{split_attr})
        node.left = left
        node.right = right
        self.buildDTree_ID3(left)
        self.buildDTree_ID3(right)

    def buildDtree_Random_attr(self, node):
        '''
        Helpers method for building
        the decision tree classifier based on random attribute selection
        '''
        if len(node.examples) == 0:
            # node doesn't have any examples
            node.output = choice([0, 1])
            self.leafs += 1
            return

        # handle leaf node cases
        if not node.attrs:
            # available attributes to split on
            # have been exhausted
            val, counts = np.unique(node.examples['Class'], return_counts=True)
            node.output = max(list(zip(counts, val)))[1]
            self.leafs += 1
            return

        if node.isSameClass():
            # node contains examples all with
            # the same class label
            node.output = node.examples['Class'][0]
            self.leafs += 1
            return

        list1 = random.sample(node.attrs, 1)
        split_attr = ''.join(list1)
        left, right = split(node, split_attr, node.attrs-{split_attr})
        node.left = left
        node.right = right
        self.buildDtree_Random_attr(left)
        self.buildDtree_Random_attr(right)


    def displayDecisionTree(self):
        '''
        Calls method to print
        the binary tree
        '''
        self.printDecisionTree(self.root.left, 0)
        self.printDecisionTree(self.root.right, 0)

    def printDecisionTree(self, node, tab):
        '''
        Prints tree using preorder binary
        tree traversal algorithm
        '''
        if not node:
            return
        # for spacing over to make tree more clear
        self.depthofleafs = self.depthofleafs + 1
        space = '|  ' * tab
        # split attribute of node
        split = node.split
        # split binary value of node
        bin_split = str(node.splitValue)
        # generate string to print
        to_print = space + split + ' = ' + bin_split + ' : '
        if node.output != None:
            # handle leaf nodes with class labels
            to_print += str(node.output)
        print(to_print)
        self.printDecisionTree(node.left, tab + 1)
        self.depthofleafs = self.depthofleafs + 1
        self.printDecisionTree(node.right, tab + 1)
        self.depthofleafs = self.depthofleafs + 1


    def labelNodes(self):
        '''
        Labels nodes using level order
        traversal. Returns node count
        and a list of leaf nodes
        '''
        queue = deque([self.root])
        count = 1
        leaf_nodes = []
        while queue:
            curr = queue.popleft()
            curr.label = count
            if not curr.left and not curr.right:
                leaf_nodes.append(count)
            count += 1
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        return count - 1, leaf_nodes


class TreeNode:

    def __init__(self, examples, attrs, parent=None, split='', name='', splitValue=None):
        # input processing
        self.examples = examples
        self.attrs = attrs
        self.label = None
        self.output = None
        self.split = split
        self.name = name
        self.splitValue = splitValue

        # branch connections
        self.parent = parent
        self.right, self.left = None, None

    def __repr__(self):
        return "{}".format(self.label)


    def getExamples(self):
        '''
        Returns the training instances
        stored in the tree node
        '''
        return self.examples

    def isSameClass(self):
        '''
        Returns true if all class labels
        are the same, otherwise returns false
        '''
        classes = self.examples['Class']
        return np.all(classes == 0) or np.all(classes)


def findEntropy(examples):
    '''
    Calculates and returns the findEntropy
    value for the examples in a data node
    '''
    ent_val = 0
    # get binary counts
    _, counts = np.unique(examples['Class'], return_counts=True)
    bin_freqs = counts.astype('float') / len(examples) 
    for freq in bin_freqs:
        if freq != 0.0:
            ent_val -= freq * np.log2(freq)
    return ent_val   


def findInformationGain(examples, attr):
    '''
    Calculates and returns the information
    gain with its corresponding attribute
    '''
    gain = findEntropy(examples)
    # get number of examples in each split
    bin_vals, counts = np.unique(examples[attr], return_counts=True)
    bin_freqs = counts.astype('float') / len(examples)
    for freq, val in zip(bin_freqs, bin_vals) :
        gain -= freq * findEntropy(examples[examples[attr] == val])
    return gain, attr


def findSplitAttribute(node):
    '''
    Using information gain, finds and
    returns the highest information gain
    value and its corresponding attribute
    given a set of attributes to split on
    '''
    examples = node.getExamples()
    parent_entropy = findEntropy(examples)
    gain_vals = [findInformationGain(examples, attr) for attr in node.attrs]
    return max(gain_vals, key=lambda x: x[0])



def split(node, split_attr, attrs):
    '''
    Splits the examples in a node 
    on the given attribute
    '''
    examples = node.getExamples()
    l = examples[examples[split_attr] == 0]
    r = examples[examples[split_attr] != 0] 
    left = TreeNode(l, attrs, parent=node, split=split_attr, splitValue=0)
    right = TreeNode(r, attrs, parent=node, split=split_attr, splitValue=1)
    return left, right


def modelDecisionTree(trainData, validData, testData, id3OrRandom):
    '''
    Print basic decision tree info as well as
    accuracy on the training and testing datasets
    '''
    DTree = ID3Algorithm(trainData, id3OrRandom)
    trainDataAcc = DTree.findAccuracy(trainData)
    validDataAcc = DTree.findAccuracy(validData)
    testDataAcc = DTree.findAccuracy(testData)
    print('-----------------------------------')
    print()
    if id3OrRandom:
        DTree.displayDecisionTree()
        print('         Random Selection Accuracy       ')
    else:
        DTree.displayDecisionTree()
        print('         ID3 algorithm Accuracy       ')
    print('-----------------------------------')
    print('Number of training instances = {}'.format(len(trainData)))
    print('Number of training attributes = {}'.format(len(DTree.attrs)))
    print('Total number of nodes in the tree = {}'.format(DTree.getNodesCount()))
    print('Number of leaf nodes in the tree = {}'.format(DTree.getLeavesCount()))
    average_depth = DTree.getDepthofLeaves() / DTree.getLeavesCount()
    print('Average Depth:{}'.format(average_depth))
    print('Accuracy of the model on the training dataset = {}'.format(trainDataAcc))
    print()

    attrs = set(list(validData.dtype.names)[:-1])
    print('Number of validation instances = {}'.format(len(validData)))
    print('Number of validation attributes = {}'.format(len(attrs)))
    print('Accuracy of the model on the validation dataset = {}'.format(validDataAcc))
    print()

    attrs = set(list(testData.dtype.names)[:-1])
    print('Number of testing instances = {}'.format(len(testData)))
    print('Number of testing attributes = {}'.format(len(attrs)))
    print('Accuracy of the model on the testing dataset = {}'.format(testDataAcc))
    print()


if __name__ == "__main__":
    trainPath, validPath, testPath = sys.argv[1:]

    # load datasets
    trainData = np.genfromtxt(trainPath, dtype='int', delimiter=',', names=True)
    validData = np.genfromtxt(validPath, dtype='int', delimiter=',', names=True)
    testData = np.genfromtxt(testPath, dtype='int', delimiter=',', names=True)
    # build decision tree and get accuracy
    # ID3
    modelDecisionTree(trainData, validData, testData, 0)
    # Random
    modelDecisionTree(trainData, validData, testData, 1)
