# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 18:45:25 2020

@author: Shrikant Agrawal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

# Splitting dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.25,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create the model for SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear',random_state=0)  # Also try Kernel = 'Poly' and then check confusion matrix. Use linear or poly based on confusion matrix results
classifier.fit(X_train, y_train)

# Predit the test set results
y_pred = classifier.predict(X_test)
    
# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm

# Check accuracy level
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

""" Till not code is similart to session 11 on SVM.
 
WE have used Kernel = linear, we can select rbf or poly. accuracy will get differ accordingly
But how we will get to know which kernel to use. Currently we can check it because we have
only two variables but in real word you will get a lot of variables and then we can not check
which are the values are linearly seperated and which are not.

So for this we apply GridSearchCV - It will provide different parameters.
Parameters means when we import any library then we initiate it. After initializing we press
shift tab and get all parameters.
GridSearchCV tell us what parameters we need to select to get best accuracy score of our mode"""


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)


accuracy = grid_search.best_score_    # This is not our accuracy score
accuracy

# Now uese this parameter in our model and compare accuracy score
grid_search.best_params_
classifier = SVC(kernel = 'rbf', gamma=0.7)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

""" Earlier accuracy was 90%, now by using correct parameters it increases to 93%






