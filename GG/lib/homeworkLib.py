from scipy.optimize import minimize
from numpy import exp, random, linalg
import numpy as np
import time
import sklearn.cluster
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot
import matplotlib.pyplot as plt

#%% MLP

class MLPNetwork():
    def __init__(self, hidden_layer_sizes, solver, sigma, rho, random_state):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = solver
        self.sigma = sigma
        self.rho = rho
        self.random_state = random_state
        self.NbrFuncEval = 0
        self.NbrGradEval = 0
        self.NormGradAtOptimalPoint = None
        self.training_time = 0
        self.number_of_inputs_per_neuron = 1

    def __random_start(self):
        #Seed the random number generator: deterministic good practice
        random.seed(self.random_state)
        #Randomly setting the layer features
        # multiplying by 2 and substracting 1 is to allow the selection of negative values
        self.synaptic_weights_hidden_layer = 2 * random.random((self.number_of_inputs_per_neuron, self.hidden_layer_sizes)) - 1
        self.synaptic_weights_output_layer = 2 * random.random((self.hidden_layer_sizes, 1)) - 1
        self.noises = 2 * random.random((1, self.hidden_layer_sizes)) - 1
        
    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x, sigma = 1):
        return (1 - exp(-sigma*x)) / (1 + exp(-sigma*x))
    
    # The neural network thinks.
    # The function does the weighted sum of the inputs
    def think(self, x, w, v, b, N, sigma):
        sum = 0
        for j in range(N):
            sum2 = 0
            for i in range (self.number_of_inputs_per_neuron):
                sum2 += w[i, j] * x[i] - b[j]
            sum += float(v[j]) * self.__sigmoid(sum2, sigma = sigma)
        return sum
    
    # Implementation of the regularized training error function E(v, w, b)
    def __error(self, omega, X, Y, N, P, sigma, rho):
        v = omega[:self.hidden_layer_sizes]
        w = omega[self.hidden_layer_sizes*2:].reshape(self.number_of_inputs_per_neuron, N)
        b = omega[self.hidden_layer_sizes:self.hidden_layer_sizes*2]
        error = 0
        for iteration in range(P):
            y_hat = self.think(X[iteration], w, v, b, N, sigma)
            y = float(Y[iteration])
            error += np.square( linalg.norm(y_hat - y) )
        error /= (2*P)
        regularization_term = rho * np.sum(np.square(w)) + rho * np.sum(np.square(b)) + rho * np.sum(np.square(v)) 
        return (error + regularization_term)

    # Implementation of the regularized training error function E(v) for extreme learning
    def __error_extreme(self, v, w, b, X, Y, N, P, sigma, rho):
        error = 0
        for iteration in range(P):
            y_hat = self.think(X[iteration], w, v, b, N, sigma)
            y = float(Y[iteration])
            error += np.square( linalg.norm(y_hat - y) )
        error /= (2*P)
        regularization_term = rho * np.sum(np.square(w)) + rho * np.sum(np.square(b)) + rho * np.sum(np.square(v)) 
        # Regularizing only v
        #regularization_term = rho * np.sum(np.square(v)) 
        return (error + regularization_term)
    
    def __error_extreme2(self, omega, v, X, Y, N, P, sigma, rho):

        w = omega[self.hidden_layer_sizes:].reshape(self.number_of_inputs_per_neuron, N)
        b = omega[:self.hidden_layer_sizes]
        
        error = 0
        for iteration in range(P):
            y_hat = self.think(X[iteration], w, v, b, N, sigma)
            y = float(Y[iteration])
            error += np.square( linalg.norm(y_hat - y) )
        error /= (2*P)
        regularization_term = rho * np.sum(np.square(w)) + rho * np.sum(np.square(b)) + rho * np.sum(np.square(v)) 
        # Regularizing only w and b
        #regularization_term = rho * np.sum(np.square(w)) + rho * np.sum(np.square(b))
        return (error + regularization_term)
    
    # We train the neural network through an optimization proccess using the input data and minimizing the error.
    def fit(self, training_set_inputs, training_set_outputs, max_number_of_training_iterations = 1000):        
        print()
        print("Optimizing the neural network...")
        # Reinitializing the number of function and gradient evaluations 
        self.NbrFuncEval = 0
        self.NbrGradEval = 0
        self.NormGradAtOptimalPoint = 0
        # Setting the number of inputs per neuron in the hidden layer
        self.number_of_inputs_per_neuron = training_set_inputs.shape[1]
        # Getting the training set size
        P = len(training_set_inputs)
        # Getting the number of neurons in the hidden layer
        N = self.hidden_layer_sizes
        # Getting the value of the regularization parameter rho to use on the global error computation
        rho = self.rho
        #getting the value of the spread in the activation function sigma
        sigma = self.sigma
        #getting the value of the solver
        solver = self.solver
        # Input data
        X = training_set_inputs
        # Output data
        Y = training_set_outputs
            
        # Preparing the initial guesses for the parameters to be minimized
        self.__random_start()
        # Computing the initial output on training data
        self.result_initial_output_train = self.predict(training_set_inputs)
        # Layer1 to layer2 weights
        v = list(np.asfarray(self.synaptic_weights_output_layer).flatten())
        # Inputs to layer1 weights
        w = list(np.asfarray(self.synaptic_weights_hidden_layer).flatten())
        # Layer1 noises
        b = list(np.asfarray(self.noises).flatten())
        
        initial_omega = np.asarray(v + b + w)
        
        start_time = time.time()
        optimized = minimize(self.__error, 
                          initial_omega, 
                          args=(X, Y, N, P, sigma, rho),
                          method = solver, 
                          options=dict({'maxiter':max_number_of_training_iterations}))
        self.training_time = time.time() - start_time
        self.NbrFuncEval = optimized.nfev
        self.NbrGradEval = optimized.njev
        self.NormGradAtOptimalPoint = linalg.norm(optimized.jac)
        result = optimized.x
        #gathering the minimization results
        v = result[:self.hidden_layer_sizes]
        w = result[self.hidden_layer_sizes*2:].reshape(self.number_of_inputs_per_neuron, self.hidden_layer_sizes)
        b = result[self.hidden_layer_sizes:self.hidden_layer_sizes*2]
        
        self.synaptic_weights_hidden_layer = w
        self.noises = b
        self.synaptic_weights_output_layer = v

    # We perform extreme learning training on the neural network through an optimization proccess using the input data and minimizing the error.
    def fit_extreme(self, training_set_inputs, training_set_outputs, max_number_of_training_iterations = 1000):        
        print()
        print("Optimizing the neural network...")
        # Reinitializing the number of function and gradient evaluations 
        self.NbrFuncEval = 0
        self.NbrGradEval = 0
        self.NormGradAtOptimalPoint = 0
        # Setting the number of inputs per neuron in the hidden layer
        self.number_of_inputs_per_neuron = training_set_inputs.shape[1]
        # Getting the training set size
        P = len(training_set_inputs)
        # Getting the number of neurons in the hidden layer
        N = self.hidden_layer_sizes
        # Getting the value of the regularization parameter rho to use on the global error computation
        rho = self.rho
        #getting the value of the spread in the activation function sigma
        sigma = self.sigma
        #getting the value of the solver
        solver = self.solver
        # Input data
        X = training_set_inputs
        # Output data
        Y = training_set_outputs
            
        # Preparing the initial guesses for the parameters to be minimized
        self.__random_start()
        # Computing the initial output on training data
        self.result_initial_output_train = self.predict(training_set_inputs)      
        # Layer1 to layer2 weights
        v = np.asfarray(self.synaptic_weights_output_layer).flatten()
        # Inputs to layer1 weights
        w = self.synaptic_weights_hidden_layer
        # Layer1 noises
        b = np.asfarray(self.noises).flatten()
                
        initial_v = v
        
        start_time = time.time()
        # In this case (Extreme learning) we could have used a non gradient based optimizer
        # since the minimization of the v's is a deterministic problem.
        optimized = minimize(self.__error_extreme, 
                          initial_v, 
                          args=(w, b, X, Y, N, P, sigma, rho),
                          method = solver, 
                          options=dict({'maxiter':max_number_of_training_iterations}))
        #optimized = least_squares(self.__error_extreme, initial_v, args = (w, b, X, Y, N, P, sigma, rho), method ='trf')
        self.training_time = time.time() - start_time
        #gathering the minimization results        
        self.NbrFuncEval = optimized.nfev
        self.NbrGradEval = optimized.njev
        self.NormGradAtOptimalPoint = linalg.norm(optimized.jac)
        #gathering the minimization results
        v = optimized.x
        self.synaptic_weights_output_layer = v

    def fit_decomposition_method(self, training_set_inputs, training_set_outputs, test_set_inputs, test_set_outputs, max_number_of_training_iterations = 1000, verbose = False):        
        print()
        print("Optimizing the neural network...")
        # Reinitializing the number of function and gradient evaluations 
        self.NbrFuncEval = 0
        self.NbrGradEval = 0
        self.NormGradAtOptimalPoint = 0
        # Setting the number of inputs per neuron in the hidden layer
        self.number_of_inputs_per_neuron = training_set_inputs.shape[1]
        # Getting the training set size
        P = len(training_set_inputs)
        # Getting the number of neurons in the hidden layer
        N = self.hidden_layer_sizes
        # Getting the value of the regularization parameter rho to use on the global error computation
        rho = self.rho
        #getting the value of the spread in the activation function sigma
        sigma = self.sigma
        #getting the value of the solver
        solver = self.solver
        # Input data
        X = training_set_inputs
        # Output data
        Y = training_set_outputs
        # Preparing the initial guesses for the parameters to be minimized
        self.__random_start()
        # Computing the initial output on training data
        self.result_initial_output_train = self.predict(training_set_inputs)
        start_time = time.time()
        # Starting the two-block decomposition method optimization of the MLP
        # Layer1 to layer2 weights
        v = np.asfarray(self.synaptic_weights_output_layer).flatten()
        # Inputs to layer1 weights
        w = np.asfarray(self.synaptic_weights_hidden_layer)
        # Layer1 noises
        b = list(np.asfarray(self.noises).flatten())
        result_outputs_test = self.predict(test_set_inputs)
        test_error = self.get_error(result_outputs_test, test_set_outputs)
        if (verbose):
            print ()
            print("Initial v: ", v)
            print()
            print("Initial w: ", w)
            print()
            print("Initial b: ", b)
            print()
            print("Initial test error: ", test_error)    
        i = 1
        flag = 1
        nfe = 0
        nje = 0
        err_count = 0
        best_error_and_iteration = [float("inf"), 0]
        while (flag and (i <= 100)):
            if (verbose):
                print()
                print("Iteration ", i)
            
            # Step1: minimization with respect to v
            optimized1 = minimize(self.__error_extreme, v, args=(w, b, X, Y, N, P, sigma, rho), method = solver, options=dict({'maxiter':max_number_of_training_iterations}))
            nfe += optimized1.nfev
            nje += optimized1.njev
            new_v = optimized1.x
            omega = np.asarray(b + list(np.asfarray(w).flatten()))
            
            # Step2: minimization with respect to c
            optimized2 = minimize(self.__error_extreme2, omega, args=(new_v, X, Y, N, P, sigma, rho), method = solver, options=dict({'maxiter':max_number_of_training_iterations}))
            nfe += optimized2.nfev
            nje += optimized2.njev
            result = optimized2.x
            
            # Update the model's weights but befor doing so we save them 
            #in order to set them back in case we are dealing with an increase in weights
            new_w = result[self.hidden_layer_sizes:].reshape(self.number_of_inputs_per_neuron, self.hidden_layer_sizes)
            new_b = result[:self.hidden_layer_sizes]
            old_w = self.synaptic_weights_hidden_layer
            old_v = self.synaptic_weights_output_layer
            old_b = self.noises
            self.synaptic_weights_hidden_layer = new_w
            self.synaptic_weights_output_layer = new_v
            self.noises = new_b
            
            # Computing the new error and comparing with the old one. we update the weights if and only if we get an improvment in error
            result_outputs_test = self.predict(test_set_inputs)
            new_test_error = self.get_error(result_outputs_test, test_set_outputs)
            
            if (verbose):
                print ()
                print("        new v: ", new_v)
                print()
                print("        new w: ", new_w)
                print()
                print("        new w: ", new_b)
                print()
                print("        new test error: ", new_test_error)
            
            # If the error increses or stayed constant for more than 4 times stop training: Early stopping method.
            if ((np.array_equal(new_v, v) and np.array_equal(new_w, w) and np.array_equal(new_b, b)) or err_count == 3):
                flag = 0
            
            # Increment this counter if the error increases or stays constant
            if (new_test_error >= test_error):
                err_count += 1
            # Reset this counter to 0 if the error decreases
            else:
                err_count = 0
            
            # Decide if the actual situation is better than the previous or not
            if (new_test_error < best_error_and_iteration[0]):
                best_error_and_iteration[0] = new_test_error
                best_error_and_iteration[1] = i
            # If it is not, put back the old weights
            else:
                self.synaptic_weights_hidden_layer = old_w
                self.synaptic_weights_output_layer = old_v
                self.noises = old_b
            
            v = new_v
            w = new_w
            b = list(new_b.flatten())
            test_error = new_test_error
            i += 1

        if (verbose):
            print ()
            print ()
            print ("Best computed error: ", best_error_and_iteration[0])
            print ()
            print ("Best computed error iteration: ", best_error_and_iteration[1])
            print ()
            
        self.training_time = time.time() - start_time
        self.NbrFuncEval = nfe
        self.NbrGradEval = nje
        self.NormGradAtOptimalPoint = linalg.norm(optimized2.jac)
        self.NbrOuterIter = i-1
        
        
    def predict(self, test_set_inputs):
        # test data
        X = test_set_inputs
        # number of records in the input test data
        P = len(test_set_inputs)
        #number of neurons of the hidden layer
        N = self.hidden_layer_sizes
        # Layer1 to layer2 weights
        v = list(np.asfarray(self.synaptic_weights_output_layer).flatten())
        # Inputs to layer1 weights
        w = self.synaptic_weights_hidden_layer
        # Layer1 noises
        b = list(np.asfarray(self.noises).flatten()) 
        y_hat = []
        for iteration in range(P):
            y_hat.append(self.think(X[iteration], w, v, b, N, self.sigma))   
        return np.asarray(y_hat).T
        
        
    # We compute the the error given the the estimated outputs and the true outputs values
    def get_error(self, test_set_outputs, test_set_true_outputs):
        y = test_set_true_outputs
        y_hat = test_set_outputs
        P = len(test_set_outputs)
        sum = 0
        for iteration in range(P):
            sum += np.square(y[iteration] - y_hat[iteration])
        return sum/(2*P)

    # The neural network prints its weights
    def print_weights(self):
        print()
        print ("    Hidden layer weights: ")
        print (self.synaptic_weights_hidden_layer)
        print()
        print ("    Hidden noises: ")
        print (self.noises)
        print()
        print ("    Output layer weights:")
        print (self.synaptic_weights_output_layer)
    
    # We implement the approximating function found by the neural net
    def approx_func(self, x, y, sigma = 1):
        #number of neurons of the hidden layer
        N = self.hidden_layer_sizes
        # Hidden to output layer weights
        v = list(np.asfarray(self.synaptic_weights_output_layer).flatten())
        # Inputs to hidden layer weights
        w = self.synaptic_weights_hidden_layer
        # Hidden layer noises
        b = list(np.asfarray(self.noises).flatten())    
        input_data = np.asarray([x, y])
        z = self.think(input_data, w, v, b, N, sigma)
        return z
    
    def plot(self):        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = y = np.arange(-2.0, 2.0, 0.05)
        X, Y = np.meshgrid(x, y)
        z = np.array([self.approx_func(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = z.reshape(X.shape)
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
        
#%% RBF
        
class RBFNetwork():
    def __init__(self, hidden_layer_sizes, solver, sigma, rho, random_state):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = solver
        self.sigma = sigma
        self.rho = rho
        self.random_state = random_state
        self.NbrFuncEval = 0
        self.NbrGradEval = 0
        self.NormGradAtOptimalPoint = 0
        self.training_time = 0
        self.number_of_inputs_per_neuron = 1

    def __random_start(self, X, P):
        #Seed the random number generator: deterministic good practice
        random.seed(self.random_state)
        #Randomly setting the layer features
        self.synaptic_weights_output_layer = 2 * random.random((self.hidden_layer_sizes, 1)) - 1
        centers_labels = random.randint(P, size=(1, self.hidden_layer_sizes))
        self.centers = X[centers_labels].reshape([self.hidden_layer_sizes, self.number_of_inputs_per_neuron])
        
    def __random_start_extreme(self, X):
        #Seed the random number generator: deterministic good practice
        random.seed(self.random_state)
        #Randomly setting the layer features
        self.synaptic_weights_output_layer = 2 * random.random((self.hidden_layer_sizes, 1)) - 1
        self.centers = sklearn.cluster.KMeans(n_clusters = self.hidden_layer_sizes).fit(X).cluster_centers_
        
    # The RBF function, in this case the Gaussian function.
    # We pass the radius (distances between the input points and the choosen centers) through this function to
    # spans the finite dimensional set where our interpolant (think) is defined.
    def __phi(self, x, sigma = 1):
        return (exp(-((x/sigma)**2)))
    
    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __phi_gradient_wrt_cj(self, x, cj, sigma):
        term1 = 2/(sigma**2)
        term2 = exp(-((linalg.norm(x - cj)/sigma)**2))
        term3 = x - cj
        return term1 * term2 * term3
    
    # The neural network thinks.
    # The function does the weighted sum of the phi's of the radius
    def think(self, x, v, c, N, sigma):
        sum = 0
        for j in range(N):
            sum += float(v[j]) * self.__phi(linalg.norm(x-c[j]))
        return sum
    
    # Implementation of the regularized training error function E(v, w, b)
    def __error(self, Vs_and_centers, X, Y, N, P, sigma, rho):
        v = Vs_and_centers[:self.hidden_layer_sizes]
        c = Vs_and_centers[self.hidden_layer_sizes:].reshape([self.hidden_layer_sizes, self.number_of_inputs_per_neuron])
        error = 0
        for iteration in range(P):
            y_hat = self.think(X[iteration], v, c, N, sigma)
            y = float(Y[iteration])
            error += np.square( linalg.norm(y_hat - y) )
        error /= (2*P)
        regularization_term = rho * np.sum(np.square(v)) + rho * np.sum(np.square(c)) 
        return (error + regularization_term)
    

    # Implementation of the regularized training error function E(v) for extreme learning
    def __error_extreme(self, v, c, X, Y, N, P, sigma, rho):
        error = 0
        for iteration in range(P):
            y_hat = self.think(X[iteration], v, c, N, sigma)
            y = float(Y[iteration])
            error += np.square( linalg.norm(y_hat - y) )
        error /= (2*P)
        regularization_term = rho * np.sum(np.square(v)) + rho * np.sum(np.square(c)) 
        #regularizing only v
        #regularization_term = rho * np.sum(np.square(v))
        return (error + regularization_term)
    
    # Implementation of the regularized training error function E(c) for two-block decomposition
    def __error_extreme2(self, c, v, X, Y, N, P, sigma, rho):
        error = 0
        for iteration in range(P):
            y_hat = self.think(X[iteration], v, c, N, sigma)
            y = float(Y[iteration])
            error += np.square( linalg.norm(y_hat - y) )
        error /= (2*P)
        regularization_term = rho * np.sum(np.square(v)) + rho * np.sum(np.square(c)) 
        # regularizing only c
        #regularization_term = rho * np.sum(np.square(c)) 
        return (error + regularization_term)
    
        
    # We train the neural network through an optimization proccess using the input data and minimizing the error.
    def fit(self, training_set_inputs, training_set_outputs, max_number_of_training_iterations = 1000):        
        print()
        print("Optimizing the neural network...")
        # Reinitializing the number of function and gradient evaluations 
        self.NbrFuncEval = 0
        self.NbrGradEval = 0
        self.NormGradAtOptimalPoint = 0
        # Setting the number of inputs per neuron in the hidden layer
        self.number_of_inputs_per_neuron = training_set_inputs.shape[1]
        # Getting the training set size
        P = len(training_set_inputs)
        # Getting the number of neurons in the hidden layer
        N = self.hidden_layer_sizes
        # Getting the value of the regularization parameter rho to use on the global error computation
        rho = self.rho
        #getting the value of the spread in the activation function sigma
        sigma = self.sigma
        #getting the value of the solver
        solver = self.solver
        # Input data
        X = training_set_inputs
        # Output data
        Y = training_set_outputs
            
        # Preparing the initial guesses for the parameters to be minimized
        self.__random_start(X, P)
        # Computing the initial output on training data
        self.result_initial_output_train = self.predict(training_set_inputs)
        # Layer2 to output layer weights
        v = list(np.asfarray(self.synaptic_weights_output_layer).flatten())
        # Centers labels
        centers = list(np.asfarray(self.centers).flatten())
        
        initial_Vs_and_centers = np.asarray(v + centers)
                       
        start_time = time.time()
        optimized = minimize(self.__error, 
                          initial_Vs_and_centers,
                          args=(X, Y, N, P, sigma, rho),
                          method = solver, 
                          options=dict({'maxiter':max_number_of_training_iterations}))
        self.training_time = time.time() - start_time
        self.NbrFuncEval = optimized.nfev
        self.NbrGradEval = optimized.njev
        self.NormGradAtOptimalPoint = linalg.norm(optimized.jac)
        result = optimized.x
        #gathering the minimization results
        v = result[:self.hidden_layer_sizes]
        c = result[self.hidden_layer_sizes:].reshape([self.hidden_layer_sizes, self.number_of_inputs_per_neuron])
        
        self.synaptic_weights_output_layer = v
        self.centers = c

    # We perform extreme learning training on the neural network through an optimization proccess using the input data and minimizing the error.
    def fit_extreme(self, training_set_inputs, training_set_outputs, max_number_of_training_iterations = 1000):        
        print()
        print("Optimizing the neural network...")
        # Reinitializing the number of function and gradient evaluations 
        self.NbrFuncEval = 0
        self.NbrGradEval = 0
        self.NormGradAtOptimalPoint = 0
        # Setting the number of inputs per neuron in the hidden layer
        self.number_of_inputs_per_neuron = training_set_inputs.shape[1]
        # Getting the training set size
        P = len(training_set_inputs)
        # Getting the number of neurons in the hidden layer
        N = self.hidden_layer_sizes
        # Getting the value of the regularization parameter rho to use on the global error computation
        rho = self.rho
        #getting the value of the spread in the activation function sigma
        sigma = self.sigma
        #getting the value of the solver
        solver = self.solver
        # Input data
        X = training_set_inputs
        # Output data
        Y = training_set_outputs
            
        # Preparing the initial guesses for the parameters to be minimized
        self.__random_start_extreme(X)
        # Computing the initial output on training data
        self.result_initial_output_train = self.predict(training_set_inputs)
        # Layer1 to output layer weights
        v = list(np.asfarray(self.synaptic_weights_output_layer).flatten())
        # Centers labels
        centers = np.asfarray(self.centers)
    
        initial_Vs = np.asarray(v)
                       
        start_time = time.time()
        # Also in this case (Extreme learning) we could have used a non gradient based optimizer
        # since the minimization of the v's is a deterministic problem.
        optimized = minimize(self.__error_extreme, 
                          initial_Vs,
                          args=(centers, X, Y, N, P, sigma, rho),
                          method = solver, 
                          options=dict({'maxiter':max_number_of_training_iterations}))
        self.training_time = time.time() - start_time
        self.NbrFuncEval = optimized.nfev
        self.NbrGradEval = optimized.njev
        self.NormGradAtOptimalPoint = linalg.norm(optimized.jac)
        #gathering the minimization results
        v = optimized.x
        self.synaptic_weights_output_layer = v


    def predict(self, test_set_inputs):
        # test data
        X = test_set_inputs
        # number of records in the input test data
        P = len(test_set_inputs)
        #number of neurons of the hidden layer
        N = self.hidden_layer_sizes
        # Output layer weights
        v = list(np.asfarray(self.synaptic_weights_output_layer).flatten())
        # Centers
        c = self.centers
        y_hat = []
        for iteration in range(P):
            y_hat.append(self.think(X[iteration], v, c, N, self.sigma))   
        return np.asarray(y_hat).T
        
        
    # We compute the the error given the the estimated outputs and the true outputs values
    def get_error(self, test_set_outputs, test_set_true_outputs):
        y = test_set_true_outputs
        y_hat = test_set_outputs
        P = len(test_set_outputs)
        sum = 0
        for iteration in range(P):
            sum += np.square(y[iteration] - y_hat[iteration])
        return sum/(2*P)

    # The neural network prints its weights
    def print_weights(self):
        print()
        print ("    Output layer weights: ")
        print (self.synaptic_weights_output_layer)
        print()
        print ("    Centers: ")
        print (self.centers)
    
    # We implement the approximating function found by the neural net
    def approx_func(self, x, y, sigma = 1):
        #number of neurons of the hidden layer
        N = self.hidden_layer_sizes
        # Hidden to output layer weights
        v = list(np.asfarray(self.synaptic_weights_output_layer).flatten())
        # Centers
        c = self.centers
        input_data = np.asarray([x, y])
        z = self.think(input_data, v, c, N, sigma)
        return z
    
    def plot(self):        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = y = np.arange(-2.0, 2.0, 0.05)
        X, Y = np.meshgrid(x, y)
        z = np.array([self.approx_func(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = z.reshape(X.shape)
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    
