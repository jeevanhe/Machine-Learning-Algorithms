# data handling libs
import numpy as np
import pandas as pd

# data preprocessing libs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# sklearn classifiers to import
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# tensorflow classifier import
import tensorflow as tf
from tensorflow.contrib.learn import DNNClassifier

# model building, predict, accuracy imports
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from IPython.display import display

# Logging level
tf.logging.set_verbosity(tf.logging.FATAL)

# Get data from csv file with headers
data = pd.read_csv("ILPD.csv", names=['age', 'gender', 'tb', 'db', 'alkphos', 'sgpt', 'sgot',
                                      'tp', 'alb', 'a/g', 'class'])

print('Number of instances in dataset:', len(data))
print('Number of attributes in dataset:', len(data.columns.values))
numfolds = 10

# categorize the "gender" - Male/Female to 0 and 1
le = LabelEncoder()
le.fit(data['gender'])
data['gender'] = le.transform(data['gender'])

# Remove any NAN rows from the dataset
data.dropna(inplace=True)

# separate feature data and target data
X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

# split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# build the parameters of all classifiers
random_forest_params = dict(n_estimators=[10, 12, 15, 20], criterion=['gini', 'entropy'],
                            max_features=['auto', 'log2', 'sqrt', None], bootstrap=[False, True]
                            )
decision_tree_params = dict(criterion=['gini', 'entropy'], splitter=['best', 'random'],
                            class_weight=['balanced', None], presort=[False, True])

perceptron_params = dict(penalty=[None, 'l2', 'l1', 'elasticnet'], shuffle=[False, True],
                         class_weight=['balanced', None])

svm_params = dict(shrinking=[False, True], class_weight=['balanced', None])

neural_net_params = dict(activation=['identity', 'logistic', 'tanh', 'relu'],
                         solver=['adam'], learning_rate=['constant', 'invscaling', 'adaptive'])

log_reg_params = dict(class_weight=['balanced', None], solver=['newton-cg', 'lbfgs', 'liblinear'])

knn_params = dict(n_neighbors=[5, 10, 12], weights=['uniform', 'distance'],
                  algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'])

bagging_params = dict(n_estimators=[10, 12, 15], bootstrap=[False, True])

ada_boost_params = dict(n_estimators=[50, 75, 100], algorithm=['SAMME', 'SAMME.R'])

guassiannb_params = dict()

gradient_boosting_params = dict(n_estimators=[100, 150, 200], loss=['deviance', 'exponential'])

params = [
    random_forest_params, decision_tree_params, perceptron_params,
    svm_params, neural_net_params, log_reg_params, knn_params,
    bagging_params, ada_boost_params, guassiannb_params, gradient_boosting_params
]
# classifiers to test
classifiers = [
    RandomForestClassifier(), DecisionTreeClassifier(), Perceptron(),
    SVC(), MLPClassifier(), LogisticRegression(),
    KNeighborsClassifier(), BaggingClassifier(), AdaBoostClassifier(),
    GaussianNB(), GradientBoostingClassifier()
]

names = [
    'RandomForest', 'DecisionTree', 'Perceptron', 'SVM',
    'NeuralNetwork', 'LogisticRegression',
    'KNearestNeighbors', 'Bagging', 'AdaBoost', 'Naive-Bayes', 'GradientBoosting'
]

models = dict(zip(names, zip(classifiers, params)))

def find_best_model_with_param_tuning(models, X_train, X_test, y_train, y_test):
    '''
    Uses grid search to find the best accuracy
    for a model by finding the best set of values
    for the parameters the model accepts
    '''
    print('How many fold cross-validation performed:', numfolds)
    accuracies = []
    # dataframe to store intermediate results
    dataframes = []
    bestparams = []
    for name, clf_and_params in models.items():
        print('Performing GridSearch on {}-classfier'.format(name))
        clf, clf_params = clf_and_params
        grid_clf = GridSearchCV(estimator=clf, param_grid=clf_params, cv=numfolds)
        grid_clf = grid_clf.fit(X_train, y_train)
        dataframes.append((name, grid_clf.cv_results_))
        bestparams.append((name, grid_clf.best_params_))
        predictions = grid_clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=numfolds)
        accuracies.append((name, accuracy, np.mean(cv_scores)))
    return accuracies, dataframes, bestparams


results, dataframes, bestparams = find_best_model_with_param_tuning(models, X_train, X_test, y_train, y_test)
print()
print('============================================================')
for classifier, acc, cv_acc in results:
    print('Classifier = {}: Accuracy = {} || Mean Cross Val Accuracy scores = {}'.format(classifier, acc, cv_acc))

for name, bp in bestparams:
    print('============================================================')
    print('{}-classifier GridSearch Best Params'.format(name))
    print('============================================================')
    display(bp)
print()
print()

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=len(X[0]))]
dl_clf = DNNClassifier(hidden_units=[10, 20, 10], n_classes=2,
                       feature_columns=feature_columns, model_dir="/tmp/ilpd")
dl_clf.fit(X_train, y_train, steps=4000)
predictions = list(dl_clf.predict(X_test, as_iterable=True))
acc = accuracy_score(y_test, predictions)
print('============================================================')
print('Classifier = {}: Accuracy = {} '.format(DNNClassifier, acc))
print('============================================================')
print('{}-classifier GridSearch Best Params'.format(DNNClassifier))
display(dl_clf.params)
print('============================================================')
