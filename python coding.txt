#load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

#load the dataset
filename = 'datafile2.csv'
names = ['sno','mens','gen','prog', 'est', 'fshlh', 'thyroid', 'prolactin', 'endo','fibrcyst','pcos','geninf','struc','pid','pao','dia','hyper','weight','stress','nutri','class' ]
dataset = pd.read_csv(filename, names=names) 
print(dataset.shape)
# set x and y
x=dataset.iloc[:,0:20].values
y=dataset.iloc[:,20].values
print("\nFirst 5 rows of X:\n", x[:5])
print("\nFirst 5 rows of y:\n", y[:41])
print(x.shape)
print(y.shape)
print(dataset.describe())
# split x and y
seed = 5
test_size = 0.20
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, random_state = seed)
# building model SVC
model = SVC()
# Training 
model.fit(xtrain,ytrain)
print(model)
# Testing
ypred = model.predict(xtest)
# Predictions
predictions = [round(value) for value in ypred]
print(predictions)
#accuracy
accuracy = accuracy_score(ytest, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Gaussian
model = GaussianNB()
model.fit(xtrain,ytrain)
print(model)
ypred = model.predict(xtest)
predictions = [round(value) for value in ypred]
print(predictions)
accuracy = accuracy_score(ytest, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Decision Tree
model = DecisionTreeClassifier()
model.fit(xtrain,ytrain)
print(model)
ypred = model.predict(xtest)
predictions = [round(value) for value in ypred]
print(predictions)
accuracy = accuracy_score(ytest, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# Gradient boosting 
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
clf.fit(xtrain, ytrain)
print(clf)
ypred = clf.predict(xtest)
predictions = [round(value) for value in ypred]
print(predictions)
accuracy = accuracy_score(ytest, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#sample testing
sample = [[1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], [1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0]]
preds = clf.predict(sample)
pred_species = [y[p] for p in preds]
print("Predictions:", pred_species)

#visualizing data
dataset.hist()
plt.show()
scatter_matrix(dataset)
plt.show()
correlations = dataset.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()




