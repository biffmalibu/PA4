'''
File: ann.py
Authors: 
    Jason Brownlee (original code, see link below)
    Hank Feild (added additional structure, functions, and TODOs for PA4)
Source: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
Purpose: Provides functions to train, run, and evaluate a multilayered perceptron neural network.
         Currently, it only supports multiclass classification.
'''

# PA4 NOTE: You may not use any libraries byond what is already provided below.
from random import randrange
from random import random
from csv import reader
from math import exp
import json


#++++++++++++++++++ Functions largely from Jason Brownlee's code ++++++++++++++++++++++++
# Load a CSV file
def load_csv(filename, skip_header=False):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)

        if skip_header:
            next(csv_reader)

        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    stats = [[minmax[0], minmax[1]] if minmax[0] != minmax[1] else [minmax[0], minmax[0] + 0.01] for minmax in stats]
    return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    dataset_copy = [row[:] for row in dataset]
    for row in dataset_copy:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset_copy

# Normalizes a training and test dataset using the minmax of the training dataset.
def normalize_train_test(train_dataset, test_dataset):
    minmax = dataset_minmax(train_dataset)
    train_dataset_normed = normalize_dataset(train_dataset, minmax)
    test_dataset_normed = normalize_dataset(test_dataset, minmax)
    return train_dataset_normed, test_dataset_normed, minmax

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs, epoch_report_n=50):
    print('Training network...', end='')
    for epoch in range(n_epoch):
        # TODO Basic 1: The code between ## START and ## END should only be executed every epoch_report_n epochs.
        if (epoch + 1) % epoch_report_n == 0:
            correct = 0
            for row in train:
                if row[-1] == predict(network, row):
                    correct += 1
            print(f'\n  Accuracy on train: {correct/len(train)*100:.2f}%')
            print(f'  Epoch {epoch}', end='')
        print('.', end='', flush=True)
        
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

    correct = 0
    for row in train:
        if row[-1] == predict(network, row):
            correct += 1
    print(f'\n  Accuracy on train: {correct/len(train)*100:.2f}%')

    print('\nTraining done!')

# Initialize a network
def initialize_network(n_inputs, n_hidden_layers, n_hidden_nodes, n_outputs):
    network = list()

    for _ in range(n_hidden_layers):
        hidden_layer = [{'weights':[random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden_nodes)]
        network.append(hidden_layer)

        n_inputs = n_hidden_nodes

    output_layer = [{'weights':[random() for _ in range(n_inputs + 1)]} for _ in range(n_outputs)]
    network.append(output_layer)
    
    return network

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

#---------------------------------------------------------------------------------------#
 

class ArtificalNeuralNetwork:
    '''An encapsulation of most of the data and functions a user would want to interact with.'''

    def __init__(self, learning_rate=0.3, epochs=500, hidden_layers=1, nodes_per_hidden_layer=5, 
                 scale_features=False, scale_min_max=None, label_to_int_lookup=None, network=None,
                 epoch_report_n=50):
        '''Initializes a new instance of the ArtificialNeuralNetwork class.
        
        Parameters:
            learning_rate (float): The learning rate for the network.
            epochs (int): The number of epochs to train the network.
            hidden_layers (int): The number of hidden layers in the network.
            nodes_per_hidden_layer (int): The number of nodes in each hidden layer.
            scale_features (bool): Whether or not to scale the features.
            scale_min_max (list): The min and max values for each column in the dataset (available as self.min_max).
            label_to_int_lookup (dict): A lookup table for converting class labels to integers.
            network (list): The network to use for predictions.
            epoch_report_n (int): The number of epochs between reports during training.
        '''
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.nodes_per_hidden_layer = nodes_per_hidden_layer
        self.scale_features = scale_features
        self.scale_min_max = scale_min_max
        self.label_to_int_lookup = label_to_int_lookup
        self.network = network
        self.epoch_report_n = epoch_report_n

    def train(self, train_data):
        '''Trains the network using the provided dataset.
        
        Parameters:
            train_data (list(list(Numeric))): A list of lists of numbers representing the dataset.
        '''
        n_inputs = len(train_data[0]) - 1
        n_outputs = len(set([row[-1] for row in train_data]))
        self.network = initialize_network(n_inputs, self.hidden_layers, self.nodes_per_hidden_layer, n_outputs)
        train_network(self.network, train_data, self.learning_rate, self.epochs, n_outputs, self.epoch_report_n)

    def eval(self, test_data):
        '''Evaluates the network using the test data.
        
        Parameters:
            test_data (list(list(Numeric))): A list of lists of numbers representing the dataset.
            
        Returns (float): The accuracy of the network on the test data.
        '''
        # Make a copy of the test data where the last column is None
        # to mask the labels.
        test_data_unlabeled = [row[:-1]+[None] for row in test_data]
        predicted = self.predict(test_data_unlabeled)
        actual = [row[-1] for row in test_data]
        return accuracy_metric(actual, predicted)
    
    def cross_validate(self, dataset, n_folds):
        '''Evaluates the network using cross validation.
        
        Parameters:
            dataset (list(list(Numeric))): A list of lists of numbers representing the dataset.
            n_folds (int): The number of folds to use for cross validation.
            
        Returns (list(float), float): A list of accuracies for each fold and the average accuracy.
        '''
        scores = []
        folds = cross_validation_split(dataset, n_folds)
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            self.train(train_set)

            test_set = fold
            scores.append(self.eval(test_set))
    
        return scores, sum(scores)/len(scores)


    def predict(self, unlabeled_data):
        '''Predicts the class labels for a dataset. Assumes there is no column for labels 
        and that this instances has already been trained (i.e., that self.network is a 
        trained network).
        
        Parameters:
            unlabeled_data (list(list(Numeric))): A list of lists of numbers representing the dataset.

        Returns (list(int)): A list of predicted class labels
        '''
        predicted = []
        for row in unlabeled_data:
            predicted.append(predict(self.network, row))
        return predicted


    def save(self, filename):
        '''Saves the network to a file in JSON format.
        
        Parameters:
            filename (str): The path to the file.
        '''
        # TODO Advanced 1: save the network and settings to a file in the JSON format. All
        # instance data members should be included.

    def load(self, filename):
        '''Loads a network from a file in the JSON format written by the save function.
        
        Parameters:
            filename (str): The path to the file.
        '''
        # TODO Advanced 2: load the network and settings from the file. This should populate
        # all of the instance data members (see __init__) by reading the file in the format
        # output by the save function.


    def load_and_process_data(self, filename, skip_header, label_to_int_lookup=None, data_has_labels=True):
        '''Loads and processes a dataset from a CSV file. Assumes the last column is the class label unless
        data_has_labels is False.
        
        Parameters:
            filename (str): The path to the CSV file.
            skip_header (bool): Whether or not to skip the header row.
            label_to_int_lookup (dict): A lookup table for converting class labels to integers.
            data_has_labels (bool): True if the last column contains labels.

        Returns (list(list(Numeric))): A list of lists of numbers representing the dataset.
        '''
        # Load data.
        dataset = load_csv(filename, skip_header)

        # Convert string columns to float.
        for i in range(len(dataset[0])-1):
            str_column_to_float(dataset, i)

        # Convert class column to integers.
        if data_has_labels:
            # If no lookup table, back up to the instance's lookup table if it exists.
            if label_to_int_lookup is None and self.label_to_int_lookup is not None:
                label_to_int_lookup = self.label_to_int_lookup

            # Create a new lookup table if necessary
            if label_to_int_lookup is None:
                self.label_to_int_lookup = str_column_to_int(dataset, len(dataset[0])-1)

            # Apply the lookup table to the last column.
            else:
                for row in dataset:
                    row[-1] = label_to_int_lookup[row[-1]]

        # Scale features.
        if self.scale_features:
            if self.scale_min_max is None:
                self.scale_min_max = dataset_minmax(dataset)
            dataset = normalize_dataset(dataset, self.scale_min_max)

        return dataset
