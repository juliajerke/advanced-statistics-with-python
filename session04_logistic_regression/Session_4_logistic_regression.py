# -*- coding: utf-8 -*-
"""
Logistic regression
@author: Julia Jerke

Date: 2021-10-14
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

from pathlib import Path


#%% Importing  the data set for the logistic regression

# Data set import
path = Path(__file__).parent / 'voting.csv'
vote = pd.read_csv(path)


#%% Data cleaning

#__ Choosing the subset of interest
vote_sub = vote[["year","bula","ostwest","q1","q2c","q2d","q7","q10","q11",
                 "q13","q17","q19aa","q19ba","q24","q32", "q33","q55","q60","q67",
                 "q73a","q73b","q73c","q73d","q73e","q73f","q78","q79","q80",
                 "q118","q133","q134", "q135","q167","q170","q172","q178","q181",
                 "q192","q136j","q136k","q136l","q136m","q137"]]

#__ Renaming the variables
vote_sub.rename({"ostwest": "west", "q1": "sex","q2c": "birthyear", "q2d": "eligible",
                 "q7": "media", "q10": "corruption", "q11": "performance",
                 "q13": "economics_state", "q17": "voted", "q19aa": "firstvote",
                 "q19ba": "secondvote", "q24": "votedpast", "q32": "leftright",
                 "q33": "democracy", "q55": "economics_own", "q60": "political_interest",
                 "q67": "chancellor", "q73a": "refugees", "q73b": "warming",
                 "q73c": "terrorism", "q73d": "globalisation", "q73e": "turkey",
                 "q73f": "nuclear", "q78": "taxes", "q79": "immigration",
                 "q80": "climate", "q118": "justice", "q133": "marital_status",
                 "q134": "partner", "q135": "school", "q167": "class",
                 "q170": "religious", "q172": "german_born", "q178": "german_born_parents",
                 "q181": "german_born_partner", "q192": "household_income",
                 "q136j": "polytechnic", "q136k": "bachelor", "q136l": "master",
                 "q136m": "doctorate", "q137": "employment"},
                axis=1,errors="raise",inplace=True)


#__ Assigning missing values
vote_sub.replace(range(-100,-50), np.nan, inplace=True)


#__ Recoding of variables

# gender
vote_sub["male"] = vote_sub.sex.replace([1,2],[1,0])
pd.crosstab(vote_sub.sex,vote_sub.male)

# calculating the age, rough approximation (we are ignoring the birth month here)
vote_sub["age"] = vote_sub.year - vote_sub.birthyear

# current voting behavior
vote_sub.voted.replace(2,0,inplace=True)
vote_sub.voted.value_counts()

# past voting behavior
vote_sub.votedpast.replace(2,0,inplace=True)
vote_sub.votedpast.value_counts()

# voted for AFD, first vote
vote_sub.firstvote.value_counts()
vote_sub["firstvote_afd"] = 0
vote_sub.loc[vote_sub["firstvote"] == 322, "firstvote_afd"] = 1
pd.crosstab(vote_sub.firstvote,vote_sub.firstvote_afd)

# voted for AFD, second vote
vote_sub.secondvote.value_counts()
vote_sub["secondvote_afd"] = 0
vote_sub.loc[vote_sub.secondvote == 322, "secondvote_afd"] = 1
pd.crosstab(vote_sub.secondvote,vote_sub.secondvote_afd)

# preference for Angela Merkel
vote_sub.chancellor.value_counts()
vote_sub["merkel"] = vote_sub.chancellor.replace([1,2,3],[1,0,0])
vote_sub.merkel.value_counts()

# partnership dummy
vote_sub.partner.value_counts()
vote_sub.partner.replace([1,2,3],[1,1,0],inplace=True)

# school education
vote_sub.school.value_counts(dropna=False)
vote_sub["education"] = vote_sub.school.replace([1,2,3,4,5,6,9],[0,0,1,2,2,np.nan,np.nan])
pd.crosstab(vote_sub.school,vote_sub.education,margins=True)

# university degree
vote_sub["university"] = np.nan
vote_sub.loc[(vote_sub["polytechnic"]==0) &
             (vote_sub["bachelor"]==0) &
             (vote_sub["master"]==0) &
             (vote_sub["doctorate"]==0),"university"] = 0
vote_sub.loc[(vote_sub["polytechnic"]==1) |
             (vote_sub["bachelor"]==1) |
             (vote_sub["master"]==1) |
             (vote_sub["doctorate"]==1),"university"] = 1
vote_sub.university.value_counts(dropna=False)



#%% Data inspection

pd.options.display.max_columns = 10 #setting the number of columns you want to display

#__ Descriptive inspection
vote_sub.info()
vote_sub.describe()

# Distribution of the variable of interest
vote_sub.secondvote_afd.value_counts()

# Possible relationship between gender and AFD voting behavior
ct = pd.crosstab(vote_sub.secondvote_afd,vote_sub.male,normalize="columns")
print(ct)


# Manually calculating the difference in probabilities
print(ct[1][1]-ct[0][1])


# Manually calculating the odds ratio of gender and AFD voting behavior
odd_female = ct[0][1] /(1-ct[0][1])
odd_male = ct[1][1] / (1-ct[1][1])
odds_ratio = odd_male/odd_female
print(odds_ratio)



#%% Simple logistic regression


#__ Describing the model
model = smf.logit("secondvote_afd ~ male", data = vote_sub)


#__ Fitting the model
results = model.fit()


#__ Summarizing the model
# Printing a comprehensive overview
print(results.summary())


# Calculating odds ratios
print(np.exp(results.params))


#__ Calculating and displaying marginal effects
marginal_ame = results.get_margeff(at="overall") # AME
print(marginal_ame.summary())

marginal_mem = results.get_margeff(at="mean") # MEM
print(marginal_mem.summary())



#__ Getting the list of all the attributes saved to the fitted model
dir(results)

#__ Displaying the AIC value (Akaike Information Criterion)
print(results.aic)


#__ Saving the predicted values to a new variable
vote_sub["predicted"] = results.predict()

predict_tab = results.pred_table(threshold=0.5) # threshold defines the classification probability



#%% Adding further sociodemographic variables


#__ Describing the model
model = smf.logit("secondvote_afd ~ male + age + C(education) + household_income", data = vote_sub)


#__ Fitting the model
results = model.fit()


#__ Summarizing the model
# Printing a comprehensive overview
print(results.summary())

# Calculating odds ratios
print(np.exp(results.params))


#__ Calculating and displaying marginal effects
marginal_ame = results.get_margeff(at="overall") # AME
print(marginal_ame.summary())

marginal_mem = results.get_margeff(at="mean") # AME
print(marginal_mem.summary())


# AIC 
print(results.aic)
