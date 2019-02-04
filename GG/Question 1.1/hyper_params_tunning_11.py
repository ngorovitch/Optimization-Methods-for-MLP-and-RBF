import sys
sys.path.insert(0, '../lib')
from homeworkLib import MLPNetwork
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# here we use a K-fold cross validation to minimize the hyperparameters    
def objective(params, k, data):
    Ns = params[0]; sigmas = params[1]; rhos = params[2]
    best_params = {'N': 0, 'sigma': 0, 'rho': 0, 'best_error': 0.02358677602613122}
    for N in Ns:
        for sigma in sigmas:
            for rho in rhos:
                print()
                print("N = ", N)
                print("sigma = ", sigma)
                print("rho = ", rho)
                MLP = MLPNetwork(hidden_layer_sizes = N, solver = 'bfgs', sigma = sigma, rho = rho, random_state = 1792126)    
                kf = KFold(n_splits=k)
                sum_error = 0
                for train, test in kf.split(data):
                    train_data = np.array(data)[train]
                    train_data = pd.DataFrame({'X':train_data[:,0],'Y':train_data[:,1], 'Z':train_data[:,2]})
                    training_set_inputs = np.asarray(train_data[['X', 'Y']])
                    training_set_outputs = np.asarray(train_data['Z']).T
                    # Train the neural network using the training set.
                    # Do it at most 60,000 times.
                    MLP.fit(training_set_inputs, training_set_outputs, 60000)
                    test_data = np.array(data)[test]
                    test_data = pd.DataFrame({'X':test_data[:,0],'Y':test_data[:,1], 'Z':test_data[:,2]})
                    test_set_inputs = np.asarray(test_data[['X', 'Y']])
                    test_set_outputs = np.asarray(test_data['Z']).T
                    # Test the neural network both on the training and the testing data
                    #result_output_train = MLP.predict(training_set_inputs)
                    result_output_test = MLP.predict(test_set_inputs)
                    sum_error += MLP.get_error(result_output_test, test_set_outputs)
                average_error = sum_error/k
                print("Test_error: ", average_error)
                if (average_error < best_params['best_error']):
                    best_params['best_error'] = average_error
                    best_params['N'] = N
                    best_params['sigma'] = sigma
                    best_params['rho'] = rho                              
    return best_params
 

#let's optimize the hyper paramiters
k = 2
data = np.asarray(pd.read_csv("../../Data/DATA.csv"))
result = objective([[16, 18],[1],[1e-5]], k, data)
best_N = result['N']; best_sigma = result['sigma']; best_rho = result['rho']; best_error = result['best_error']
print()
print("best_N = ", best_N)
print("best_sigma = ", best_sigma)
print("best_rho = ", best_rho)
print("best_error = ", best_error)