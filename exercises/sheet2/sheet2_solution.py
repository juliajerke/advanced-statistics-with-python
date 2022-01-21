# -*- coding: utf-8 -*-
"""
Solution for exercise sheet 2

@author: Julia Jerke
Date: 22.10.2021

"""
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from pathlib import Path


#%% Task 1 - Importing  the data set 

# Data set import
path = Path(__file__).parent / 'voting.csv'
vote = pd.read_csv(path)


#%% Task 2a - Data cleaning


#__ Choosing the subset of interest
# interdauer_ges was added to the list
vote_sub = vote[["year","bula","ostwest","q1","q2c","q2d","q7","q10","q11",
                 "q13","q17","q19aa","q19ba","q24","q32", "q33","q55","q60","q67",
                 "q73a","q73b","q73c","q73d","q73e","q73f","q78","q79","q80",
                 "q118","q133","q134", "q135","q167","q170","q172","q178","q181",
                 "q192","q136j","q136k","q136l","q136m","q137","intdauer_ges"]]

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
vote_sub.secondvote.value_counts(dropna=False)
vote_sub["secondvote_afd"] = np.nan
vote_sub.loc[vote_sub.secondvote == 322, "secondvote_afd"] = 1
vote_sub.loc[vote_sub.secondvote.isin([1,4,5,6,7,801]), "secondvote_afd"] = 0
pd.crosstab(vote_sub.secondvote,vote_sub.secondvote_afd,margins=True)
vote_sub.secondvote_afd.value_counts()


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



#%% Task 2b and 2c - Data cleaning

# Filtering all observations that have only 10 or fewer missing values
# Here, we have to pay attention that we recoded some (n=7) variables so that we 
# have now seven more variables than initially. These additional variables have 
# to be ignored when dropping cases with many missings.

shape = vote_sub.shape
miss = shape[1] -7 - 10 # calling the number of variables and substracting the threshold

# with "tresh" we can set the number of non-missing values (i.e. valied values) required
# with "subset" we can determine a subset of variables within which python should check
# for missing values, using that paramter we can excluded the 7 recoded variables from the analysis
vote_clean = vote_sub.dropna(thresh=miss,subset=["west", "sex","birthyear", 
                                                 "eligible","media", "corruption", 
                                                 "performance","economics_state", 
                                                 "voted", "firstvote","secondvote", 
                                                 "votedpast", "leftright", "democracy", 
                                                 "economics_own", "political_interest",
                                                 "chancellor", "refugees", "warming",
                                                 "terrorism", "globalisation", "turkey",
                                                 "nuclear", "taxes", "immigration",
                                                 "climate", "justice", "marital_status",
                                                 "partner", "school", "class",
                                                 "religious", "german_born", "german_born_parents",
                                                 "german_born_partner", "household_income",
                                                 "polytechnic", "bachelor", "master",
                                                 "doctorate", "employment"]) 
vote_clean.shape 

# Dropping the 10% fastest respondents
fast10 = vote_clean.intdauer_ges.quantile(0.1)
print(fast10)

vote_clean = vote_clean[vote_clean.intdauer_ges > fast10]
vote_clean.shape



#%% Task 3 - Data inspection

# Plotting a histogram of the Left-Right self-assessment
plt.hist(vote_clean.leftright)

# Getting descriptive statistics
vote_clean.leftright.describe()



#%% Task 4 - Relationship between variables

# Age
vote_clean[["age","leftright"]].corr()
plt.scatter(vote_clean.age,vote_clean.leftright)
# Comment: The relationship between age and left-right is positive but very small, 
# also the scatter does not really help since the left-right variable is not continuos (overplotting)

# Gender
vote_clean.groupby("male").leftright.describe() # complete overview
vote_clean.groupby("male").leftright.mean() # only the mean values
# Comment: male respondents lean slighty more to the right end of the scale 

# east vs. west
vote_clean.groupby("west").leftright.describe()
vote_clean.groupby("west").leftright.mean()
# Comment: respondents from West Germany lean slighty more to the right end of the scale 


# education
vote_clean.groupby("education").leftright.describe()
vote_clean.groupby("university").leftright.describe()
# Comment: The higher the education the smaller the  value on the left-right scale
# respondents with a university degree lean more to the left end of the scale

vote_clean.leftright.corr(vote_clean.education, method="spearman")

#%% Task 5 - Regressin with socio-demographic information

# Running the regression and printing the results
model = smf.ols("leftright ~ age + male + west ", data=vote_clean)
results = model.fit()
print(results.summary())
# west: respondents from West Germany have 0.48 more points to the right corner 
#   than respondents from East Germany (ceteris paribus)
# male: male respondents have 0.45 more points on the self-assesment to the right 
#   than female respondents (ceteris paribus)
# age: ceteris paribus, with each year in age the score on the left-right 
#   self-assessments increases by 0.009 points to the right corner
# education=1: respondents with intermediate educational degree have 0.002 more 
#   points on the scale than respondents with no or low education (ceteris paribus)
# education=2: reponsdents with high education lean 0.48 more points on the 
#   self-assessment scale to the LEFT than respondents with no or low education (c.p.)

# Note: except for education=1, all coefficients are significant


# R-square
print(results.rsquared)
# Comment: The r square is 0.051, hence only 5.1% of the variation (=variance) 
#   in the left-right self-assessment is explained by the variables in the model

# Repeat the regression with an interaction term of age and gender
model = smf.ols("leftright ~ age*male + west + C(education)", data=vote_clean)
results = model.fit()
print(results.summary())
# Comment: the effect of age on the left-right self-assessment is 0.007 points 
#   smaller for male respondents, ceteris paribus; however, the effect is not significant


#%% Task 6 - Including political attitudes in the analysis

# Correlation between the fear variables and Left-Right self-assessment
pd.options.display.max_columns = 10 #ignore that for the moment!
vote_clean[["leftright","refugees", "warming", "terrorism", "globalisation",
            "turkey", "nuclear"]].corr()


# Regression analysis including political attitudes
model = smf.ols("""leftright ~ age*male + west + C(education) + refugees 
                      + warming + terrorism + globalisation + turkey + nuclear""", data=vote_clean)
results = model.fit()
print(results.summary())
# refugees: if the worries with respect to the refugee crisis increase by one unit, 
#   the left-right self-assessment increases by 0.264, ceteris paribus
# warming: if the worries with respect to the global warming increase by one unit, 
#   the left-right self-assessment decreases by 0.113, ceteris paribus
# terrorism: if the worries with respect to terrorism increase by one unit, 
#   the left-right self-assessment increases by 0.144, ceteris paribus
# globalisation: if the worries with respect to the globalisation increase by one unit, 
#   the left-right self-assessment decreases by 0.044, ceteris paribus
# turkey: if the worries with respect to the developments in Turkey increase by one unit, 
#   the left-right self-assessment decreases by 0.071, ceteris paribus
# nuclear: if the worries with respect to nuclear power increase by one unit, 
#   the left-right self-assessment decreases by 0.147, ceteris paribus
# Except for globalisation all coefficents are significant at the 5 percent level

print(results.rsquared)    
# Comment: the r square is now 0.159, the model notably improved by adding the variables to the model
# Now, the model can explain 15.9% of the variance of the left-right self-assessment


# Saving the predicted values an the residuals into new variables
vote_clean["predicted"] = results.fittedvalues
vote_clean["residuals"] = results.resid


# Regression diagnostics
plt.hist(results.resid, bins=50)
results.resid.describe()
# Comment: the mean of the residuals is approximately 0 and the histogram suggests 
#   that they are approximately normal distributed



#%% Task 7 a and b - Dummy recoding

# Corruption
vote_clean.corruption.value_counts(dropna=False)

# long way...
vote_clean["corruption_dummy"] = np.nan
vote_clean.loc[vote_clean.corruption.isin([1,2]), "corruption_dummy"] = 1
vote_clean.loc[vote_clean.corruption.isin([3,4]), "corruption_dummy"] = 0
pd.crosstab(vote_clean.corruption_dummy,vote_clean.corruption,margins=True)
vote_clean.corruption_dummy.value_counts(dropna=False)

# short way...
vote_clean["corruption_dummy"] = vote_clean.corruption.replace([1,2,3,4], [1,1,0,0])


# Satisfaction with the government
vote_clean.performance.value_counts(dropna=False)
vote_clean["performance_dummy"] = np.nan
vote_clean.loc[vote_clean.performance.isin([3,4]), "performance_dummy"] = 1
vote_clean.loc[vote_clean.performance.isin([1,2]), "performance_dummy"] = 0
pd.crosstab(vote_clean.performance_dummy,vote_clean.performance,margins=True)
vote_clean.performance_dummy.value_counts(dropna=False)

vote_clean["performance_dummy"] = vote_clean.performance.replace([1,2,3,4], [0,0,1,1])


#%% Task 7 c and d - Regression analysis including the satisfaction with the current situation in the country

model = smf.ols("""leftright ~ age*male + west + C(education) + refugees 
                      + warming + terrorism + globalisation + turkey + nuclear
                      + corruption_dummy + performance_dummy""", data=vote_clean)
results = model.fit()
print(results.summary())
# corruption_dummy: respondents that belief that there is (widespread) corruption 
#   score 0.161 points higher towards the right corner (ceteris paribus)
# performance_dummy: respondents that attest the government a bad performance 
#   score 0.17 points more towards the LEFT corner (ceteris paribus)
# However, both coefficients are not significant at the common significance level

print(results.rsquared)
# Comment: the r square is 0.173, hence there is only a minor improvement in exlanatory power



#%% Task 8 - Saving the data set

vote_clean.to_csv("Voting_clean.csv")
