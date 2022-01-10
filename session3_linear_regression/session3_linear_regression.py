# -*- coding: utf-8 -*-
"""
Linear regression
@author: Julia Jerke

Date: 2021-10-07
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from pathlib import Path

#__ You can also manually define the path in which you are working
# Defining the working directory
# Uncomment the two lines below and replace ... with your path
#from os import chdir
#chdir(...)


#%% Importing and preparing the data set

#__ Importing the World Happiness data set
path = Path(__file__).parent / 'world_happiness_2019.csv'
happy = pd.read_csv(path)


#__ Renaming the variables
# Transforming all variable names to lower case
happy.rename(str.lower, axis=1, inplace=True)
# Renaming the variables to simpler names
happy.rename(columns={"overall rank": "rank",
                      "country or region": "country", 
                      "gdp per capita": "gdp",
                      "social support": "support",
                      "healthy life expectancy": "life",
                      "freedom to make life choices": "choices",
                      "perceptions of corruption": "corruption"},
          inplace=True, errors="raise")

#%% Importing the continent information

path_continent = Path(__file__).parent / 'country_continent.xlsx'

continent = pd.read_excel(path_continent)
# Transforming all variable names to lower case
continent.rename(str.lower, axis=1, inplace=True)


#__ Joining the continent information to the happiness data set
happy = happy.join(continent.set_index("country"), on="country")

# Note: develop = 1 for developing countries


#%% Data inspection

#__ Descriptive inspection
pd.options.display.max_columns = 10 #setting the number of columns you want to display
happy.info()
happy.describe()


#__ Visual inspection (more on that in a later session)
plt.scatter(happy.gdp, happy.score)
plt.xlabel("GDP per capita")
plt.ylabel("Happiness score")
plt.show()
     


#%% Simple linear regression


#__ Describing the model
model = smf.ols("score ~ gdp", data = happy)


#__ Fitting the model
results = model.fit()


#__ Summarizing the model
# Printing a comprehensive overview
print(results.summary())
# Printing just the regression coefficients
print(results.params)


#__ Getting the list of all the attributes saved to the fitted model
dir(results)


#__ Saving the predicted values to a new variable
happy["predicted"] = results.fittedvalues


#__ Saving the error terms to a new variable
happy["resid"] = results.resid


#__ Getting to the confidence interval of the predicted regression line is a bit more complicated

# Call a more comprehensive prediction summary
prediction = results.get_prediction()

# Confidencde interval
# Access the lower value of the CI
ci_lower = prediction.summary_frame()["mean_ci_lower"]
# Access the upper value of the CI
ci_upper = prediction.summary_frame()["mean_ci_upper"]


# Prediction interval
# Access the lower value of the CI
pred_lower = prediction.summary_frame()["obs_ci_lower"]
# Access the upper value of the CI
pred_upper = prediction.summary_frame()["obs_ci_upper"]


# Save the two values of the CI into a new data frame and append it to our existing data frame
conf_int = pd.DataFrame({"lower": ci_lower,"upper": ci_upper})
happy = happy.join(conf_int)


#%% Testing the assumption of the regression model

#__ Are the residuals normally distributed with mean 0?
plt.hist(happy.resid) # don't worry, we will go into more detail in the session on visualization
plt.show()
plt.close()


#__ Do we have homoscedasticity?
plt.scatter(happy.predicted,happy.resid)
plt.show()
plt.close()


#%% Plotting the regression results

plt.scatter(happy.gdp, happy.score)
plt.plot(happy.gdp,happy.predicted)
plt.plot(happy.gdp,happy.lower)
plt.plot(happy.gdp,happy.upper)
plt.xlabel("GDP per capita")
plt.ylabel("Happiness score")
plt.show()


#%% Linear regression with categorical variables


model_cat = smf.ols("score ~ gdp + develop + C(continent)"
                    , data=happy)
results_cat = model_cat.fit()

print(results_cat.summary())



#%% Linear regression with interaction terms


model_inter = smf.ols("score ~ gdp*develop", data=happy)
results_inter = model_inter.fit()

print(results_inter.summary())



#%% Linear regression with quadratic term


model_quad = smf.ols("score ~ gdp + I(gdp**2)", data=happy)
results_quad = model_quad.fit()

print(results_quad.summary())

fitted = pd.Series(results_quad.fittedvalues, name="fitted")

happy_quad = happy.join(fitted)


plt.close("all")
plt.scatter(happy_quad.gdp, happy_quad.score)
plt.scatter(happy_quad.gdp,happy_quad.fitted)
plt.show()

#%%