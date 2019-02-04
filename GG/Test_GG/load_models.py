import sys
sys.path.insert(0, '../lib')
import pandas as pd
import numpy as np
import pickle
from homeworkLib import MLPNetwork, RBFNetwork

data_set = pd.read_csv("../../Data/DATA.csv")

# Randomly spliting the data
training_set=data_set.sample(frac=0.75, random_state=1792126)
training_set_inputs = np.asarray(training_set[['X', 'Y']])
training_set_outputs = np.asarray(training_set['Z']).T

test_set=data_set.drop(training_set.index)
test_set_inputs = np.asarray(test_set[['X', 'Y']])
test_set_outputs = np.asarray(test_set['Z']).T

# Fetch the neural net from the disk
with open("../pickle/neural_net_best.file", "rb") as f:
    NN = pickle.load(f)

# Test the neural network both on the training and the testing data
result_output_train = NN.predict(training_set_inputs)
result_output_test = NN.predict(test_set_inputs)
 
print()
print("Number of neurons N: ", NN.hidden_layer_sizes)
print()
print("Initial Training Error: ", NN.get_error(NN.result_initial_output_train, training_set_outputs))
print()
print("Final Training Error: ", NN.get_error(result_output_train, training_set_outputs))
print()
print("Final Test Error: ", NN.get_error(result_output_test, test_set_outputs))
print()
print("Optimization solver chosen: ", NN.solver)
print()
print('Norm of the gradient at the optimal point: ', NN.NormGradAtOptimalPoint)
print()
print("Time for optimizing the network(in seconds): ", NN.training_time)
print()
print("Number of function evaluations: ", NN.NbrFuncEval)
print()
print("value of sigma: ", NN.sigma)
print()
print("Value of rho: ", NN.rho)
print()
#print("Number of outer iteretions: ", NN.NbrOuterIter)
#print()
print("Other hyperparameters: ", None)
print()
print("Plot of the approximating function found: ")
print()
NN.plot()