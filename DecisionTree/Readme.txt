Assignment 2
CS6375 - Machine Learning
Names: Jeevan Hunsur Eswara & SriHarshareddy Munjuluru
===========================================================================================================================
Contents of the partii folder:
1) data - datasets
2) Results - output obtained by running ID3 algorithm on two datasets
3) Screenshots - Execution and result snippets
4) Report - Brief decription and learings
5) DecisionTree_ID3.py - python file for program execution
6) Readme file

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