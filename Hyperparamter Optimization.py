#hyperparameters considered are iterations, alpha, hidden layer number, nodes in the hidden layer
import math
import numpy as np
import random
import GeneralNeuralNetwork
import time
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
random.seed(time.time())

def network_evaluation(accuracy,Time):
    try:
        first = (1/3) * math.log(accuracy - (1/3),10) + 0.058697
    except:
        first = -float("inf")
    return first - (Time / 2) ** 2

def node_comb(layers,nodes):
    choices = []
    for i in combinations_with_replacement(nodes,layers):
        choices.append(list(i))
    return choices

def grid_search(inputs,outputs,Input,Output,subsections,X,Y,fast=True):
    try:
        Begin = time.time()
        x_values = []
        y_values = []
        alphas = [i / 1000 for i in range(1,1001,int(1000/subsections))]
        hidden_layers = [0,1,2,3,4] if fast == False else [0,1,2]
        hidden_nodes = [i for i in range(inputs,outputs+1)] if inputs <= outputs else [i for i in range(inputs,outputs-1,-1)]
        random.shuffle(hidden_nodes)
        hidden_nodes = hidden_nodes [:subsections]
        #iterations = [i for i in range(1,10001,int(1000/subsections))]
        iteration = 1000
        track = [0,0]
        optimal = -float("inf")

        for alpha in alphas:
            print(alpha)
            for layer in hidden_layers:
                for hid_layers in node_comb(layer,hidden_nodes):
                    print(hid_layers)
                    if hid_layers == []:
                        hid_layers = [0]
                    Time = 0
                    accuracy = 0
                    for i in range(3):
                        initial = time.time()
                        Trained = GeneralNeuralNetwork.no_momentum_train(inputs,hid_layers,outputs,iteration,Input,Output,alpha)
                        Time += time.time() - initial
                        Count = 0
                        for j in range(len(X)):
                            result = Trained.forwardprop(np.array([X[i]]))
                            if int(round(result[0][0])) == Y[i]:
                                    Count += 1
                        accuracy += Count / len(X)
                    Time = Time / 3
                    accuracy = accuracy / 3
                    if network_evaluation(accuracy,Time) > optimal:
                        optimal = network_evaluation(accuracy,Time)
                        x_values.append(time.time() - Begin)
                        y_values.append(optimal)
                        print(optimal)
                        print(x_values)
                        print(y_values)
                        track[0] = alpha
                        track[1] = hid_layers
        print(x_values)
        print(y_values)
        plt.title("Optimality of Solution Found Over Time: Grid Search Algorithm")
        plt.xlabel('Time Passed (s)')
        plt.ylabel('Evaluation Score')
        plt.plot(x_values,y_values,"xb-")
        plt.show()
        return track
    except:
        print(x_values)
        print(y_values)
        plt.title("Optimality of Solution Found Over Time: Grid Search Algorithm")
        plt.xlabel('Time Passed (s)')
        plt.ylabel('Evaluation Score')
        plt.plot(x_values,y_values,"xb-")
        plt.show()
        return track

def bin_search(inputs,outputs,Input,Output,X,Y,fast=True):
    try:
        Begin = time.time()
        x_values = []
        y_values = []
        
        iteration = 1000
        track = [0,0]
        optimal = -float("inf")
        hid_layers = [8]
        hi = 1
        low = 0

        while True:
            med2 = (hi + low) / 2
            med3 = (hi + med2) / 2
            med1 = (low + med2) / 2
            values = [med1,med2,med3]
            results = [0,0,0]
            for k in range(3):
                Time = 0
                accuracy = 0
                for i in range(3):
                    initial = time.time()
                    Trained = GeneralNeuralNetwork.no_momentum_train(inputs,hid_layers,outputs,iteration,Input,Output,values[k])
                    Time += time.time() - initial
                    Count = 0
                    for j in range(len(X)):
                        result = Trained.forwardprop(np.array([X[i]]))
                        if int(round(result[0][0])) == Y[i]:
                                Count += 1
                    accuracy += Count / len(X)
                Time = Time / 3
                accuracy = accuracy / 3
                results[k] = accuracy
            if results[0] < results[1] and results[1] < results[2]:
                low = med2
            elif results[0] < results[1] and results[1] >= results[2]:
                low = med1
                hi = med3
            elif results[0] >= results[1] and results[1] < results[2]:
                if results[2] > results[0]:
                    low = med2
                else:
                    hi = med2
            elif results[0] >= results[1] and results[1] >= results[2]:
                hi = med2
            alpha = values[results.index(max(results))]
            print(alpha)
            Eval = network_evaluation(max(results),Time)
            if Eval > optimal:
                optimal = Eval
                x_values.append(time.time() - Begin)
                y_values.append(optimal)
                track[0] = alpha
                track[1] = hid_layers
        plt.title("Optimality of Solution Found Over Time: Binary Search Algorithm")
        plt.xlabel('Time Passed (s)')
        plt.ylabel('Evaluation Score')
        plt.plot(x_values,y_values,"xb-")
        plt.show()
        return track
    except:
        print(x_values)
        print(y_values)
        plt.title("Optimality of Solution Found Over Time: Binary Search Algorithm")
        plt.xlabel('Time Passed (s)')
        plt.ylabel('Evaluation Score')
        plt.plot(x_values,y_values,"xb-")
        plt.show()
        return track


                    
                    

