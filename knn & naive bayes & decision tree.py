# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

#Entering dateset
df = pd.read_csv("C:/Users/zahra/OneDrive/Desktop/dota2.csv")
df.head()
Z= df[['r-hero1','r-hero2','r-hero3','r-hero4','r-hero4','d-hero1','d-hero2','d-hero3','d-hero4','d-hero5','avg_mmr','num_mmr','game_mode','lobby_type','radiant_win']]
X= df[['r-hero1','r-hero2','r-hero3','r-hero4','r-hero4','d-hero1','d-hero2','d-hero3','d-hero4','d-hero5','avg_mmr','num_mmr','game_mode','lobby_type']]
y= df['radiant_win']

#Spliting datas into train and test
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y,test_size=0.2)

#==============================================================================

#Knn Algorithm

#Normalizing datas
from sklearn import preprocessing
a = X_train1.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
a_scaled = min_max_scaler.fit_transform(a)
X_train1 = pd.DataFrame(a_scaled)

a = X_test1.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
a_scaled = min_max_scaler.fit_transform(a)
X_test1 = pd.DataFrame(a_scaled)

#Training and predicting
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train1, y_train1)
y_pred = classifier.predict(X_test1)

#Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print(accuracy_score(y_test1,y_pred))
print(confusion_matrix(y_test1, y_pred))
print(classification_report(y_test1, y_pred))

#Comparing Error Rate with the K Value
error = []
r = range(1, 51, 5)
for i in r:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train1, y_train1)
    pred_i = knn.predict(X_test1)
    error.append(np.mean(pred_i != y_test1))
    plt.figure(figsize=(12, 6))
    plt.plot(error, color='purple', linestyle='dashed', marker='o',
         markerfacecolor='navy', markersize=8)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    
    
#=============================================================================

#Naive bayes Algorithm

#Normalizing datas
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train1 = sc.fit_transform(X_train1)
X_test1 = sc.transform(X_test1)

#Training and predicting
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train1, y_train1)

y_pred = classifier.predict(X_test1)

#Evaluating
from sklearn.metrics import confusion_matrix, accuracy_score
print(accuracy_score(y_test1,y_pred))
print(confusion_matrix(y_test1, y_pred))
print(classification_report(y_test1, y_pred))

#=============================================================================

#Decision Tree 

#Training and predicting
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train1, y_train1)
y_pred = classifier.predict(X_test1)

#Evaluating
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print(accuracy_score(y_test1,y_pred))
print(confusion_matrix(y_test1, y_pred))
print(classification_report(y_test1, y_pred))


    

