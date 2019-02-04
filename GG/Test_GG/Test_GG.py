import sys
sys.path.insert(0, '../lib')
import pickle
import numpy as np
import pandas as pd
from homeworkLib import MLPNetwork

def run_me(X, Y):
    # Fetch the best neural net from the disk
    with open("../pickle/neural_net_best.file", "rb") as f:
        neural_network = pickle.load(f)
    y_hat = np.asarray(neural_network.predict(np.asarray(X)))
    y = np.asarray(Y)
    return neural_network.get_error(y_hat, y)

if __name__ == "__main__":
    '''# Code that create the best model and save it on the disk
    N = 16; sigma = 1; rho = 1e-5; solver = 'bfgs'

    # Creating the MLP
    MLP = MLPNetwork(hidden_layer_sizes = N, solver = solver, sigma = sigma, rho = rho, random_state = 1792126)

    # In the training set, We have 200 examples, each consisting of 2 input values
    # and 1 output value.
    data_set = pd.read_csv("../../Data/DATA.csv")

    # Randomly spliting the data 
    training_set=data_set.sample(frac=0.75, random_state=1792126)
    training_set_inputs = np.asarray(training_set[['X', 'Y']])
    training_set_outputs = np.asarray(training_set['Z']).T
    
    test_set=data_set.drop(training_set.index)
    test_set_inputs = np.asarray(test_set[['X', 'Y']])
    test_set_outputs = np.asarray(test_set['Z']).T

    MLP.fit_decomposition_method(training_set_inputs, training_set_outputs, test_set_inputs, test_set_outputs, 60000, verbose = True)
    
    # Save the Neural network on the disk
    with open("../pickle/neural_net_best.file", "wb") as f:
        pickle.dump(MLP, f, pickle.HIGHEST_PROTOCOL)'''
    

    