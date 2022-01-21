# -*- coding: utf-8 -*-
"""
Getting and exploring data with Pandas
@author: Julia Jerke

Date: 2021-09-30
"""

#%% 1 - Importing Packages

#__ First, we have to import pandas! 
# "as pd" defines an abbreviation by which we will refer to Pandas throughout the script
import pandas as pd
# we also import numpy, a library for mathematical computations and algebra
import numpy as np
# Further we add the Path function from the pathlib module that we use 
# to access the directory of this script
from pathlib import Path


#%% 2 - Creating a series

#__ We pass a list of integer numbers and store it in s
# Note, how we use the abbreviation "pd" that we defined in the beginning 
s = pd.Series([23,24,35,19,27,27,30,22,31])
print(s)


#__ We can also define labels for the index, basically it is the name for the rows
# Take care that you define as many index labels as you have rows
t = pd.Series([23,24,35,19,27,27,30,22,31], 
              index = ["a","b","c","d","e","f","g","h","i"])
print(t)


# Example of how to create a series that has stored the same value in each row
# range(0,9) creates a sequence from 0 to 9
u = pd.Series(1.0, index = range(0,10))
print(u)


# We can also create a series that has stored a sequence of numbers
# Note, that we gave the series a name in this example
v = pd.Series(range(0,9), name="numbers")
print(v)



#%% 3 - Vectorized operations with series

#__ Operations are executed per row
v_double = v+v
print(v_double)

v_square = v*v
v_square = v_square.rename(index="squared numbers") # this is how you can add a name or rename the series
print(v_square)


#__ Operations with different series
v_two = v_double + v_square
print(v_two)


#__ Adding a costant value to all entries is also possible
v_sum = v + 10
print(v_sum)



#%% 4 - Creating a DataFrame

#__ Creating a data frame from scratch
# Basis is usually a dictionary
df1 = pd.DataFrame({"name": ["Anna", "Paul", "Sarah", "Max", "Michael"],
                    "age": [23, np.nan , 23, 27, 25],
                    "program": ["BA", "MA", "MA","BA","MA"]})
print(df1)
# Set an existing variable as the index label (row names)
df1.set_index("name",inplace=True)
print(df1)


#__ Creating a data frame from already existing series
numbers = pd.Series(range(1,26))
squares = numbers*numbers
df2 = pd.DataFrame({"numbers": numbers, "squares": squares})


#__ Difference between series and data frame: a data frame has labelled columns
# Compare the output:
print(numbers)
print(pd.DataFrame(numbers, columns=["numbers"]))



#%% 5a - Getting data in and out I


#__ You can easily save your data frame to a csv or excel file
# csv:
df1.to_csv("students_overview.csv")
# excel:
df1.to_excel("students_overview.xlsx")


#%% 5b - Getting data in and out II


#__ With the same logic you can import data from a csv or excel file
# csv:
df1_csv = pd.read_csv("students_overview.csv")
print(df1_csv)
# excel:
df1_excel = pd.read_excel("students_overview.xlsx", 
                          index_col="name", 
                          usecols=["name","age","program"])
print(df1_excel)


#%% 6 - Getting data in and out III


#__ Let's again import the data set from the Covid survey and work with it
# the command Path(__file__).parent calls the directory/location of this script
# for this to work, make sure that the script and the csv file are in the same folder
cs_path = Path(__file__).parent / 'COVID-19-Survey-Student-Responses.csv'
cs = pd.read_csv(cs_path)
#Look at the variable explorer and inspect the data set
# We can set the ID as the index label:
cs = pd.read_csv(cs_path, index_col="ID")


#__ Inspecting the data frame
#pd.options.display.max_columns = 10 
cs.head(20) # See the first 20 rows
cs.shape # returns the numbers of columns and rows
cs.info() # Get some information about the dimensions and the columns (=variables) of your data frame



#%% 7a - Creating subsets: column selection 

#__ Selecting a specific column as a single series
cs["Stress busters"]
stress = cs["Stress busters"]
print(stress)

#__ Selecting more than one column, thereby creating a new (smaller) data frame
# Note, that you now have to insert the variable names as a list with brackets: [...]
cs_subset1 = cs[["Age of Subject","Time spent on Online Class",
                "Change in your weight","Stress busters"]]
cs_subset1.shape
# Creating a subset by dropping columns
cs_subset2 = cs.drop(columns=["Time spent on TV", "Time utilized"])
cs_subset2.shape


#__ The inplace parameter
# inplace = False: the original data frame IS NOT changed
# inplace = True: the original data frame IS changed
cs_inplace = cs
cs_inplace.shape
cs_inplace.drop(columns=["Time spent on TV", "Time utilized"], 
                inplace = False)
cs_inplace.drop(columns=["Time spent on TV", "Time utilized"], 
                inplace = True)



#%% 7b - Creating subsets: case selection and filtering

#__ Selecting a random sub-sample
# Setting a proportion to be selected randomly:
cs_50percent = cs.sample(frac = 0.5)
cs_50percent.shape
# Setting the number of cases to be selected randomly:
cs_100cases = cs.sample(n = 100)
cs_100cases.shape


#__ Selecting cases that meet a certain condition
cs["Age of Subject"].describe()
# Select subjects who are older than the median
age_above_median = cs[cs["Age of Subject"] > 20] # adding in brackets the condition that cases have to meet
age_above_median["Age of Subject"].describe()
# Select cases whose age is between the first and the third quartile (inclusive)
age_between_quartiles = cs[(cs["Age of Subject"]>=17) 
                           & (cs["Age of Subject"]<=21)]
age_between_quartiles["Age of Subject"].describe()



#%% 8 - Data manipulation

#__ Changing variable names
# Columns which labels are not changed stay as they are
# Parameter inplace: save results into a new variable if "False" or directly change the data frame if "True"
# Parameter errors: raise an error if a key is not found in the list of columns if "raise"
cs.rename(columns={"Region of residence": "region", 
                   "Age of Subject": "age"},
          inplace=True, errors="raise")
cs.info()


#__ Changing values
# Be careful: values that are not specified in the dictionary will be changed into NaN (=missing)
cs["Medium for online class"].value_counts()
cs["Medium for online class"] = (cs["Medium for online class"].map({
    "Laptop/Desktop": "PC", 
    "Smartphone": "Phone", "Any Gadget": "Other", 
    "Smartphone or Laptop/Desktop": "PC or Phone",
    "Tablet": "Tablet"},
    na_action = "ignore")
)


#__ Defining a new column based on existing columns
# Initialize a new variable by naming a new column name in brackets
# Calculations will be element-wise in the rows
cs["fitness_week"] = cs["Time spent on fitness"] * 7
cs["fitness_week"].describe()
cs["Time spent on fitness"].describe()

# We can check whether there are cases for which the time variables sum up to more than 24 hours
cs["time_total"] = (cs["Time spent on Online Class"] +
                    cs["Time spent on self study"] +
                    cs["Time spent on fitness"] +
                    cs["Time spent on sleep"] +
                    cs["Time spent on social media"]
)

cs["time_total"].describe()
cs_clean = cs[cs["time_total"]<=24] # filter rows that have a time sum below 24
cs_clean.shape


#__ Recoding of variables
# First, create a new variable by appending the new name in brackets to the data frame
cs["fitness_cat"] = 0
# Second, subsequently define the values for your new variable
# Using the "loc" function we can localize rows that meet a certain condition
# In brackets, we first specify the specific rows and then (separetd by a comma) the column of interest
# Using the equal sign we can then assign a new value to the specified rows
cs.loc[cs["fitness_week"] > 0, "fitness_cat"] = 1
cs.loc[cs["fitness_week"] > 5, "fitness_cat"] = 2
cs.loc[cs["fitness_week"] > 10, "fitness_cat"] = 3


#%% 9 - Missing value treatment

# Our sample has 1182 cases
cs.info()

#__ Drop rows that have at least one missing value
covid_nomiss = cs.dropna(how="any") # any is the default
covid_nomiss.info() # 1131 cases left


#__ Drop rows that have a missing value in each column
covid_nomiss = cs.dropna(how="all") 
covid_nomiss.info() # 1182 cases left



#%% 10 - Basic data description

pd.options.display.max_columns = 10 #ignore that for the moment!


#__ Frequencies
cs["Number of meals per day"].value_counts()
#cs["Number of meals per day"].value_counts(normalize=True,dropna=False)


#__ Bivariate frequencies: crosstab
pd.crosstab(cs["fitness_week"],cs["fitness_cat"])
#pd.crosstab(cs["fitness_week"],cs["fitness_cat"], normalize=True)


#__ Summary statistics
# Print basic summary statistics of the numerical variables
# These operations generally exclude missing values
# Use "describe" to get an overview
cs.describe(include="all")
cs["Number of meals per day"].describe()
# Mean
cs.mean()
# Median
cs.median()


#__ Summary statistics in groups
# The "groupby" command groups the data into categories
# Every operation is then applied to the groups
cs.groupby("region").mean() # mean by region for all variables
cs.groupby("region")["age"].mean() # variant 1
cs[["region","age"]].groupby("region").mean() # variant 2


#__ Bivariate correlations
cs[["Age of Subject",
    "Time spent on Online Class",
    "Time spent on fitness"]].corr()
cs[["Age of Subject",
    "Time spent on Online Class",
    "Time spent on fitness"]].corr(method="spearman")

#%%
