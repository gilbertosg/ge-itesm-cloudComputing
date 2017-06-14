# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 11:20:09 2016

@author: Gilbeto Silva

Download data from kaggle
https://www.kaggle.com/c/titanic/data

VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)

SPECIAL NOTES:
Pclass is a proxy for socio-economic status (SES)
 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

Age is in Years; Fractional if Age less than One (1)
 If the Age is Estimated, it is in the form xx.5

With respect to the family relation variables (i.e. sibsp and parch)
some relations were ignored.  The following are the definitions used
for sibsp and parch.

Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
Parent:   Mother or Father of Passenger Aboard Titanic
Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

Other family relatives excluded from this study include cousins,
nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
only with a nanny, therefore parch=0 for them.  As well, some
travelled with very close friends or neighbors in a village, however,
the definitions do not support such relations.

"""


#%% Python native libraries

# import os

#%% 3-rd party libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from scipy.special import expit
from sklearn import tree 
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals.six import StringIO
import pydot_ng

#%% Change script working directory to script's directory

# os.chdir("D:/Users/jonathan/Desktop/ITESM - Santa Fe")

#%% Fit a decision tree with the titanic data

def fit_titanic_decision_tree(df, features, max_depth=2):
    X = df[features].values
    Y = df['Survived'].values
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, Y)
    pred = clf.predict(X)
    return clf, Y, pred

#%% Read the data

df = pd.read_csv('train.csv')

#%% Try to fit a decision tree with raw data
features = df.columns[2:]

# First attempt
clf, Y, pred = fit_titanic_decision_tree(df, features, max_depth=2)

#%% See data types for each column
df.dtypes
df.head(10)

# Remove columns that are obviously irrelevant
df.drop(['PassengerId','Name', 'Ticket'], axis=1, inplace=True)

#%% Remove incomplete columns

def missing_data_by_cols(df):
    """
    Returns a pandas data frame with the columns and the percentage of missing data for each column.
    """
    missing = 100 - np.array([df[c].notnull().sum()*100.0 / df.shape[0] for c in df.columns])
    return pd.DataFrame({'Column':df.columns, 'Missing %':missing})

missing_data = missing_data_by_cols(df)

#%% Remove 'Cabin' column as it contains missing information

df.drop(['Cabin'], axis=1, inplace=True)

#%% Fill missing values using Imputer (basically replace anything by the mean)

""" 
NOTE: Set axis=0 when imputing more than one column so the mean is computed 
for each column. In this special case of imputing one column, a vector
is sent to the imputer, which is read as one row with many columns.
"""
imp = Imputer(missing_values=np.nan, strategy='mean', axis=1)
df['Age'] = imp.fit_transform(df['Age'].values).T


#%% Fill missing data using groups mean

df[['Sex','Age']].groupby('Sex').mean() # Means by group

# Fill each group's NA's with the group mean
f = lambda x: x.fillna(x.mean())
df['Age'] = df[['Sex','Age']].groupby('Sex').transform(f)

#%% Remove any rows having missing observations

df.dropna(axis=0, inplace=True)

#%% Missing data after cleansing

missing_data = missing_data_by_cols(df)
missing_data

#%% Data before normalization

numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
df[numerical_features].boxplot()

#%% Normalize numeric data (remove mean and divide by std. dev. each column)

scaler = StandardScaler()
df.loc[:,['Age', 'SibSp', 'Parch', 'Fare']] = scaler.fit_transform(df[numerical_features])

#%% Visualize numerical features after standardization

df[numerical_features].boxplot()

#%% Softmax normalization. Expit = sigmoid = logistic function

df.loc[:,numerical_features] = expit(df[numerical_features].values)

#%% Visualize numerical features after softmax normaliaztion

df[numerical_features].boxplot()

#%% Categorical Data - Transform each label to a binary column

def binarize_label_columns(df, columns):
    binlabel_names = []
    lb_objects = {}
    for col in columns:
        rows_notnull = df[col].notnull() # Use only valid feature observations
        lb = LabelBinarizer()
        binclass = lb.fit_transform(df[col][rows_notnull]) # Fit & transform on valid observations
        lb_objects[col] = lb
        if len(lb.classes_) > 2:
            col_binlabel_names = [col+'_'+str(c) for c in lb.classes_]
            binlabel_names += col_binlabel_names # Names for the binarized classes
            for n in col_binlabel_names: df[n] = np.NaN # Initialize columns
            df.loc[rows_notnull, col_binlabel_names] = binclass # Merge binarized data
        else: 
            binlabel_names.append(col+'_bin') # Names for the binarized classes
            df[col+'_bin'] = np.NaN # Initialize columns
            df.loc[rows_notnull, col+'_bin'] = binclass # Merge binarized data
    return df, binlabel_names, lb_objects

label_columns = ['Pclass','Embarked','Sex']
df, binlabel_features, lb_objects = binarize_label_columns(df, label_columns)

df.drop(['Pclass','Embarked','Sex'], axis=1, inplace=True)


#%% Fit a classifier with the clean dataset

features = numerical_features + binlabel_features
clf, Y, pred = fit_titanic_decision_tree(df, features, max_depth=2)

# Show performance for each class
precision_recall_fscore_support(Y, pred, average=None)

#%% Save tree to pdf file

dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data, feature_names=features)
graph = pydot_ng.graph_from_dot_data(dot_data.getvalue())  
graph.write_pdf("titanic.pdf")

#%% Label binarization step-by-step

# First 10 observations
df.head(10)

# Binarize discrete labels
lb = LabelBinarizer()

# Data contains invalid observations
df['Embarked'].unique()

# Fit & transform on valid observations
binclass = lb.fit_transform(df['Embarked']) 
lb.classes_

# Add new columns for the binarized classes
binclass_names = ['Embarked_'+str(c) for c in lb.classes_]
for n in binclass_names: df[n] = np.NaN 
df.columns

# Merge binarized data
df.loc[:, binclass_names]
df.loc[:, binclass_names] = binclass
df.loc[:, binclass_names]

# See if everything went well
embarked_cols = ['Embarked'] + binclass_names
df.head(10)[embarked_cols]
