# Implement the backpropagation algorithm for Neural Networks. and
# test it on various real world datasets. Before running the algorithm, you will have to ensure that
# all the features of the dataset are properly scaled, standardized, and categorical variables are
# encoded using numerical values.
# The two steps of this assignment are listed below. You have to create separate class for each of
# the steps.
# 1. Pre-processing:
# Pre-processing involves checking the dataset for null or missing values, cleansing the dataset of
# any wrong values, standardizing the features and converting any nominal (or categorical)
# variables to numerical form. This step is essential before running neural net algorithm, as they
# can only accept numeric data and work best with scaled data.
# The arguments to this part will be:
# - complete input path of the raw dataset
# - complete output path of the pre-processed dataset
# Your pre-processing code will read in a dataset specified using the first command line argument
# and first check for any null or missing values. You will remove any data points (i.e. rows) that
# have missing or incomplete features.
# Then, you will perform the following for each of the features (independent variables):
# - If the value is numeric, it needs to be standardized, which means subtracting the mean from
# each of the values and dividing by the standard deviation.
# See here for more details: https://en.wikipedia.org/wiki/Feature_scaling#Standardization
# - If the value is categorical or nominal, it needs to be converted to numerical values.
# For example, if the attribute is gender and is encoded as "male" or "female", it needs to be
# converted to 0 or 1. You are free to figure out specifics of your encoding strategy, but be sure to
# mention it in the report.
# For the output or predicted variable, if its value is numerical or continuous, you are free to
# convert it into categorical or binary variables (classification problem) or keep it as it is
# (regression problem). You have to specify this clearly in your report.
# After completing these steps, you need to save the processed dataset to the path specified by
# the second command line argument and use it for the next step.
# 2. Training a Neural Net:
# You will use the processed dataset to build a neural net. The input parameters to the neural net
# are as follows:
# - input dataset – complete path of the post-processed input dataset
# - training percent – percentage of the dataset to be used for training
# - error tolerance – acceptable value of error i.e. the value of error metric at which the algorithm
# can be terminated
# - number of hidden layers
# - number of neurons in each hidden layer
# For example, input parameters could be:
# ds1 80 0.01 2 2 2
# The above would imply that the dataset is ds1, the percent of the dataset to be used for
# training is 80%, the error tolerance is 0.01, and there are 2 hidden layers with (2, 2) neurons.
# Your program would have to initialize the weights randomly. Remember to take care of the bias
# term (w0) also.
# While coding the neural network, you can make the following assumptions:
# - the activation function will be sigmoid
# - you can use the backpropagation algorithm described in class and presented in the textbook
# - the training data will be randomly sampled from the dataset. The remaining will form the test
# dataset
# - you can use the mean square error as the error metric
# - one iteration involves a forward and backward pass of the back propagation algorithm
# - you can set a limit on the maximum number of iterations your algorithm will run. That is, your
# algorithm will terminate when either the error tolerance is met or the max number of iterations
# is done.
# After building the model, you will output the model parameters as below:
# Hidden Layer1:
# Neuron1 weights:
# Neuron 2 weights:
# ..
# Hidden Layer2:
# Neuron1 weights:
# Neuron 2 weights:
# ..
# Output Layer:
# Neuron1 weights:
# Neuron 2 weights:
# ..
# Total training error = ….
# You will also apply the model on the test data and report the test error:
# Total test error = ….
# Testing your program
# You will test both parts of your program, first the pre-processing part and then the model
# creation and evaluation, on the following datasets:
# 1. Boston Housing Dataset
# https://archive.ics.uci.edu/ml/datasets/Housing
# 2. Iris dataset
# https://archive.ics.uci.edu/ml/datasets/Iris
# 3. Adult Census Income dataset
# https://archive.ics.uci.edu/ml/datasets/Census+Income

import sys
import numpy as np
from numpy import *


class NeuronLayer():

    def __init__(self, number_of_inputs_per_neuron,number_of_neurons):
        self.synaptic_weights = 2 * random.random(( number_of_neurons,number_of_inputs_per_neuron)) - 1
        

class NN():
    
    def __init__(self,layer):
        self.layer = layer
        
    def nonlin(x, deriv=False):
        if (deriv == True):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
    def printvalues(self):
        print(self.layer.synaptic_weights.shape)
    
    def errorfunction(self,output,typeofdata):
        if(typeofdata =="Train"):
            print("Training error output : ")
            errormeasure = extracted_data-output
        if (typeofdata == "Test"):
            print("Test error output : ")
            errormeasure = extracted_testoutput - output
        sumerror = sum(errormeasure)
        FinalError = (1 / (2 * len(errormeasure))) * sumerror 
        if (FinalError < 0):
            print(-FinalError)
            return -FinalError
        else:
            print(FinalError)
            return FinalError


    def train(self,data0,extracted_data,originallayercount,minval,typeofdata):
        firstiteration = 0
        weights_list = []
        for iteration in range(2):
            # Forward Pass
            print("######################################")
            print("Iteration " + str(iteration+1) )
            print("######################################")
            the_list = []
            originaldata=data0
            the_list.clear()
            count =0
            weightsname = "weights"+str(count)
            dataname = "data" + str(count)
            dataname = data0
            if(firstiteration==0):
                weightsname = self.layer.synaptic_weights
                weights_list.append(weightsname)
            layercount= originallayercount
            while (layercount != 0):
                value = np.dot(dataname, weights_list[count].T)
                outputname ="output"+str(count)
                count=count+1
                outputname= nonlin(value)
                dataname = "data"+str(count)
                dataname = outputname
                num_rows,num_cols = outputname.shape
                if(layercount-1!=0 and firstiteration==0):
                    minval = minval + 1
                    numberofnuerons = sys.argv[minval]
                    layername = NeuronLayer(int(num_cols), int(numberofnuerons))
                    weightsname = "weights"+str(count)
                    weightsname = layername.synaptic_weights
                    weights_list.append(weightsname)
                if(layercount-1==0):
                    print("================================================")
                    print("Output Layer")
                    print("================================================")
                else:
                    print("================================================")
                    print("Hidden Layer " + str(count) )
                    print("================================================")
                print(dataname)
                the_list.append(dataname)
                layercount=layercount-1

            LengthofList = len(the_list)
            countval=1
            listactlength = LengthofList
            lengthofweight = LengthofList-2
            errorlist=[]
            if(typeofdata=="Train"):
               self.errorfunction(the_list[LengthofList-1],"Train")
            
            else:
                Errormeasure_test= self.errorfunction(the_list[LengthofList - 1], "Test")

            while(LengthofList!=0):
                if(LengthofList == len(the_list)):
                    l2_delta = (extracted_data- the_list[LengthofList-1]) * (the_list[LengthofList-1] *(1-the_list[LengthofList-1]))
                    weights_list[LengthofList-1] += the_list[LengthofList - 2].T.dot(l2_delta).T

                elif(LengthofList ==1):
                    l1_delta = l2_delta.dot(weights_list[LengthofList]) * (the_list[LengthofList - 1] * (1 - the_list[LengthofList - 1]))
                    weights_list[LengthofList-1] += originaldata.T.dot(l1_delta).T
                else:
                    l1_delta = l2_delta.dot(weights_list[LengthofList]) * ( the_list[LengthofList-1]* (1 - the_list[LengthofList-1]))
                    weights_list[LengthofList-1] += the_list[LengthofList-2 ].T.dot(l1_delta).T
                    l2_delta= l1_delta
                countval=countval+1
                LengthofList= LengthofList-1
                lengthofweight=lengthofweight-1
            #print("WeightsList of Length:"+ str(len(weights_list)))
            firstiteration=1

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":

    # Fetch the system variables
    random.seed(1)
    inputPath = sys.argv[1]
    perctoread = sys.argv[2]
    errorlimit = sys.argv[3]
    layernumber = sys.argv[4]
    argcount = int(layernumber)
    minval =5
    data = np.genfromtxt(inputPath, delimiter=",", skip_header=1)
    num_rows, num_col = data.shape
    rowsperc = int(int(num_rows) * (int(perctoread) / 100))
    finalrows = int(num_rows) - rowsperc
    data1 = np.genfromtxt(inputPath, delimiter=",", usecols=range(0, num_col - 1), skip_header=int(finalrows))
    testdata = np.genfromtxt(inputPath, delimiter=",", usecols=range(0, num_col - 1), skip_header=int(rowsperc))
    data2 = np.genfromtxt(inputPath, delimiter=",", skip_header=int(finalrows))
    extracted_data = data2[:, [num_col - 1]]
    data3 = np.genfromtxt(inputPath, delimiter=",", skip_header=int(rowsperc))
    extracted_testoutput= data3[:, [num_col - 1]]
    numberofnuerons = sys.argv[minval]
    layername = NeuronLayer(int(num_col - 1), int(numberofnuerons))
    neuralnetwork = NN(layername)
    print("Train data")
    neuralnetwork.train(data1,extracted_data,argcount,minval,"Train")
    print("Test data")
    neuralnetwork.train(testdata, extracted_testoutput, argcount, minval,"Test")
    