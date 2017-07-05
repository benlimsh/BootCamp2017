import pandas as pd
from pydataset import data
import numpy as np
import matplotlib.pyplot as plt

#Problem 1
diamonds = data("diamonds")
cut = diamonds.groupby("cut")
means = cut.mean()
errors = cut.std()

means.loc[:,["price"]].plot(kind="bar", yerr=errors, title = "Mean Diamond Data by Quality")
plt.xlabel("Diamond cut classification")
plt.ylabel("Price")

means.loc[:,["table"]].plot(kind="bar", yerr=errors, title = "Mean Diamond Data by Quality")
plt.xlabel("Diamond cut classification")
plt.ylabel("Table")

plt.show()
'''
'''

#Problem 2
titanic = pd.read_csv('/Users/benjaminlim/Documents/BootCamp2017/Labs/Computation/Week 2/titanic.txt')
meanage = np.mean(titanic['Age'])
titanic["Age"] = titanic["Age"].fillna(meanage).astype(float)
titanic.dropna(axis = 0)
print(titanic)

age = pd.cut(titanic['Age'], [0,12,18,80])

print(titanic.groupby('Embarked').mean()['Survived'])
print(titanic.pivot_table('Survived', index = ['Embarked'], columns = 'Sex', aggfunc = 'mean', fill_value = 0.0))
print(titanic.pivot_table('Survived', index = ['Embarked'], columns = ['Sex', age], aggfunc = 'mean', fill_value = 0.0))
print(titanic.pivot_table('Survived', index = ['Embarked', 'Pclass'], columns = ['Sex', age], aggfunc = 'mean', fill_value = 0.0))
'''
'''
