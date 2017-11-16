print("Program start!")

from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.cross_validation import cross_val_score
import pandas as pd
import numpy

logreg = LogisticRegression()

#Import data
traindata = pd.read_csv('traindata.csv', header=None)
trainlabel = pd.read_csv('trainlabel.csv', header=None)
testdata = pd.read_csv('testdata.csv', header=None)

#Transform datatype trainlabel from dataframe to array)
trainlabela = trainlabel.values
trainlabelb = trainlabela.ravel()

print("Mean of cross validation score with cv=10 using logistic regression:")
print(cross_val_score(logreg, traindata, trainlabelb, cv=10, scoring='accuracy').mean())

#Train model
logreg.fit(traindata, trainlabelb)

#Predict result from test data
testlabel=logreg.predict(testdata)

#Export result
df = pd.DataFrame(testlabel)
df.to_csv("Project1_55768.csv", index=False, header=False)

print("Result exported!")

print("Program completed!")
