# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:16:23 2023

@author: ceren
"""

import pandas as pd
import numpy as np
import seaborn as sns

#Veriseti İnceleme
df = pd.read_csv("Hotel Reservations.csv", sep = ",")
df.head()

df.info()

df.drop('Booking_ID',axis='columns',inplace=True)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype=='object':
        df[col]=label_encoder.fit_transform(df[col])

x= df.drop('booking_status',axis=1)
y= df['booking_status']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20 ,random_state=10)

print(x_train.shape)  
print(x_test.shape)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression() 

lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)

print('Test Accuracy of logistic regression: {:.2f}'.format(lr.score(x_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_predict)
print(confusion_matrix)
sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, fmt='.2%', cmap='Blues')

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
