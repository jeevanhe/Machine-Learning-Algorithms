
Execution Steps: Language used for this program is Python
[This code works only in python3]

We have DecisionTree_ID3.py. In order to execute this file first install library for numpy

		pip3 install numpy

Navigate to path where the DecisionTree_ID3.py in command prompt using: 
		
		CD <path to python file>

To run this program, after all of the conditions are met, you will run the following command:
		
		python DecisionTree_ID3.py <path-to-training-data> <path-to-validation-data> <path-to-testing-data> pruning-factor

		Ex: python DecisionTree_ID3.py training_set.csv validation_set.csv test_set.csv 0.05

  
Note: if the pruning factor is greater than number of leaf nodes program throws an error message : 'prune factor is too large'