import sys
sys.path.insert(0, '../lib')
import numpy as np
from homeworkLib import RBFNetwork
import pandas as pd
import pickle

if __name__ == "__main__":
    # The hyper params where choosing by using a Grid Search Cross Validation method: hyper_params_tunning.py
    # bfgs refers to the Broyden-Fletcher-Goldfarb-Shanno algorithm
    N = 50; sigma = 1; rho = 1e-5; solver = 'bfgs'

    # Creating the RBF
    RBF = RBFNetwork(hidden_layer_sizes = N, solver = solver, sigma = sigma, rho = rho, random_state = 1792126)

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

    # Train the neural network using the training set.
    # Do it at most 60,000 times.
    RBF.fit(training_set_inputs, training_set_outputs, 60000)
    
    # Test the neural network both on the training and the testing data
    result_output_train = RBF.predict(training_set_inputs)
    result_output_test = RBF.predict(test_set_inputs)
 
    print()
    print("Number of neurons N: ", N)
    print()
    print("Initial Training Error: ", RBF.get_error(RBF.result_initial_output_train, training_set_outputs))
    print()
    print("Final Training Error: ", RBF.get_error(result_output_train, training_set_outputs))
    print()
    print("Final Test Error: ", RBF.get_error(result_output_test, test_set_outputs))
    print()
    print("Optimization solver chosen: ", solver)
    print()
    print('Norm of the gradient at the optimal point: ', RBF.NormGradAtOptimalPoint)
    print()
    print("Time for optimizing the network(in seconds): ", RBF.training_time)
    print()
    print("Number of function evaluations: ", RBF.NbrFuncEval)
    print()
    print("value of sigma: ", sigma)
    print()
    print("Value of rho: ", rho)
    print()
    print("Other hyperparameters: ", None)
    print()
    print("Plot of the approximating function found: ")
    print()
    RBF.plot()
    
    # Save the Neural network on the disk
    with open("../pickle/neural_net_12.file", "wb") as f:
        pickle.dump(RBF, f, pickle.HIGHEST_PROTOCOL)
        
    
    
    