import numpy as np
import math
import random
import time
random.seed(time.time())

def sigmoid(x,deriv=False):
        if deriv == True:
            return sigmoid(x) * (1 - sigmoid(x))
        return 1 / (1 + math.e ** (-x))

def cost_function(output, dersired):
        return sum(0.5 * (desired - output) ** 2)

class neural_network():
    def __init__(self,in_size,hid_layer,out_size):        
        self.in_size = in_size
        self.out_size = out_size
        self.hid_layer = hid_layer
        self.weights = [0 for i in range(len(hid_layer)+1)]

        self.weights[0] = np.random.randn(self.in_size,self.hid_layer[0]) / np.sqrt(self.in_size)
        self.weights[-1] = np.random.randn(self.hid_layer[-1],self.out_size) / np.sqrt(self.hid_layer[-1])

        for i in range(1,len(self.weights)-1):
            self.weights[i] = np.random.randn(self.hid_layer[i-1],self.hid_layer[i]) / np.sqrt(self.hid_layer[i-1])
        
    def forwardprop(self, Input):
        self.sums = [0 for i in range(len(self.hid_layer)+2)]
        self.results = [0 for i in range(len(self.hid_layer)+2)]
        self.sums[0] = Input
        self.results[0] = Input
        for i in range(len(self.hid_layer)+1):
            self.sums[i+1] = self.results[i].dot(self.weights[i])
            self.results[i+1] = sigmoid(self.sums[i+1])
        return self.results[-1]
        
    def dcost_dweight(self, Input, desired):
        self.out_result = self.forwardprop(Input)

        delta = [0 for i in range(len(self.weights))]
        dW = [0 for i in range(len(self.weights))]

        delta[-1] = np.multiply(-(desired - self.out_result), sigmoid(self.sums[-1],True))
        dW[-1] = np.dot(self.results[-2].T, delta[-1])
        
        for i in range(-2,-len(delta)-1,-1):
            delta[i] = np.dot(delta[i+1],self.weights[i+1].T) * sigmoid(self.sums[i],True)
            dW[i] = np.dot(self.results[i-1].T,delta[i])
        
        return dW

def no_momentum_train(inputs,hiddens,outputs,iterations,Input,Output,alpha):
    network = neural_network(inputs,hiddens,outputs)
    for i in range(iterations):
        #neural network accuracy and time through each iteration
        #print(i)
        change = network.dcost_dweight(Input,Output)
        for j in range(len(change)):
            network.weights[j] -= alpha * change[j]
    return network

def momentum_train(inputs,hiddens,outputs,iterations,Input,Output,alpha,mu):
    network = neural_network(inputs,hiddens,outputs)
    V = [0 for i in range(len(network.weights))]
    for i in range(iterations):
        change = network.dcost_dweight(Input,Output)
        for j in range(len(change)):
            V[j] = mu * V[j] - alpha * change[j]
            network.weights[j] += V[j]
    return network
