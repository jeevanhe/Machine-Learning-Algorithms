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
    