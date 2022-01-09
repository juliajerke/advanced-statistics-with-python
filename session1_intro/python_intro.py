# -*- coding: utf-8 -*-
"""
Introduction to Python and Spyder

@author: Julia Jerke

Date: 2021-09-23
"""


#%% Block 1 - Importing packages

# import an entire package
import pandas as pd 
import numpy as np

# import a specific function from a package
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression

#%% Block 2 - Creating variables and printing

a = 7  # Variable assignment can easily be done by using "="
b = 5  

c = a + b # arithmetic operations are fairly simple in Python

# If you want to print something, you need the print() command
c
print("The result is: " + str(c))



#%% Block 3 - Importing  data into a data frame and inspecting it briefly

# The pandas library provides an easy way to import csv files, but also other file formats can be imported
path = Path(__file__).parent / 'COVID-19-Survey-Student-Responses.csv'
df = pd.read_csv(path)


#%%

# After importing that data set you can explore it in the Variable Explorer

pd.options.display.max_columns = 10 #ignore that for the moment!

# Have a look at the first 15 rows of the data
print(df.head(15))

# Get some information about the dimensions of your data frame
print(df.info())

# Print basic summary statistics of the numerical variables
print(df.describe())


#%% Block 4 - Inspecting specific variables

# Only print the summary statistics of one variable by selecting it from the data frame
print(df['Age of Subject'].describe())

# Only calculate the mean, but this time of two variables: selecting it as a list from the data frame
print(df[['Age of Subject', 'Time spent on fitness']].mean())

# Count the frequency of a categorical variable
print(df['Change in your weight'].value_counts())

# Aggregate statistics over groups of another variable
print(df[['Time spent on fitness','Change in your weight']].groupby('Change in your weight').mean())



#%% Block 5 - Producing a simple scatter plot

# Defining the variables on the axes
X1 = df['Age of Subject']
Y1 = df['Time spent on fitness']

# Create a figure object
fig1 = plt.figure()

# Defining the scatter plot
plt.scatter(X1,Y1) 

# Assigning title and axis labels
plt.title('Relationship between age and sport')
plt.xlabel('Age of subjects')
plt.ylabel('Amount of sport per week (hours)')

# Display the figure (equivalent to the 'print' command)
plt.show()

# Save the figure as a pdf
fig1.savefig('My_scatter.pdf') 


#%% Block 6 - Producing a another scatter plot

# Defining the variables on the axes
X2 = df['Time spent on self study']
Y2 = df['Time spent on social media']

# Create a figure object
fig2 = plt.figure()

# Defining the scatter plot
plt.scatter(X2,Y2) 

# Assigning title and axis labels
plt.title('Relationship between self study and social media behavior')
plt.xlabel('Time spent on self study (hours)')
plt.ylabel('Time spent on social media (hours)')

# Display the figure (equivalent to the 'print' command)
plt.show()

# Save the figure as a pdf
fig2.savefig('My_scatter2.pdf') 

#%% Block 7 - A linear regression

x = np.array(df['Time spent on self study'])
x = x.reshape(-1,1)
y = np.array(df['Time spent on social media'])

model = LinearRegression()
model.fit(x,y)

print('Intercept: ', model.intercept_)
print('Coefficient: ', model.coef_[0])
print('R-square: ', model.score(x, y))

#%%