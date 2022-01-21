# -*- coding: utf-8 -*-
"""
Solution for exercise sheet 3

@author: Julia Jerke
Date: 08.12.2021

"""
#%%

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pathlib import Path


#%% Task 1 - Importing  the data set 

# Data set import
path = Path(__file__).parent / 'voting.csv'
vote = pd.read_csv(path)



#%% Task 2a - Data cleaning


#__ Choosing the subset of interest
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


# Filtering all observations that have only 10 or fewer missing values
shape = vote_sub.shape
miss = shape[1] -7 - 10 # calling the number of variables and substracting the threshold

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

# Dropping the 10% fastest respondents
fast10 = vote_clean.intdauer_ges.quantile(0.1)
print(fast10)

vote_clean = vote_clean[vote_clean.intdauer_ges > fast10]
print(vote_clean.shape)



#%% Task 2a - Data inspection

# How many people didn't vote?
voting = vote_clean.voted.value_counts()
non_vote = vote_clean.voted.value_counts()[0]
print(str(non_vote) + " people didn't vote.")



#%% Task 2b - Relationship between voting behavior and socio-demographic characteristics

# Voting behavior and age
vote_clean.groupby("voted").age.describe()
# The respondents that did not vote seem to be younger on average.

# Voting behavior and gender
vote_clean.groupby("voted").male.value_counts()
pd.crosstab(vote_clean.voted,vote_clean.male,normalize="columns",margins=True)
# The share of non-voters is roughly the same for both gender in the data set
pd.crosstab(vote_clean.voted,vote_clean.male,normalize="index",margins=True)
# The share of male and female respondents among the voters/non-voters is more or less the same


# Voting behavior and education
pd.crosstab(vote_clean.voted,vote_clean.education,normalize="index",margins=True)
# The share of respondents with higher education is higher among voters than among non-voters
# Likewise, the share of respondents with no or low education is much higher among
#   non-voters than among voters


#%% Task 3 - logistic regression with socio-demographic variables


#__ Describing the model
model_1 = smf.logit("voted ~ male + age + C(education)", data = vote_clean)

#__ Fitting the model
results_1 = model_1.fit()

#__ Summarizing the model
# Printing a comprehensive overview
print(results_1.summary())
# As expected, education as well as age have a significant effect on the voting behavior
# The gender does not significantly affect the voting behavior


# 3b.Interpretation of the coefficients

# Since the interpretation is based on the Log odds, we can only make a statement 
#   about the direction of the effect, hence we will have a look at the sign of 
#   the coefficients:
# Gender: Being male increases the probability of voting (ceteris paribus); 
#   however, the coefficient is not significant
# Age: An increase in age also increases the probability of voting (c.p.)
# Education (1 vs. 0): A medium level of education seems to increase the probability of voting 
#   compared to no or low education (c.p.)
# Education (2 vs. 0): A high level of education seems to increase the probability of voting 
#   compared to no or low education (c.p.)


# 3c. Calculating odds ratios
print(np.exp(results_1.params))


# 3d. Exemplary interpretation of the odds ratios (OR)

# Education (2 vs. 0): 
# Log odd = 1.97: Having a high education increases the log odd of voting by 1.97 
#   compared to no or low education
# Odds ratio = 7.16: Having a high education increases the odds of voting 
#   (i.e. the chance of voting compared to the chance of not voting) 
#   by a factor of 7.16 compared to no or low education
# However, for both interpretations the problems remains that we cannot really interpret 
#   the effect in the sense of a change in probabilities


# 3e. OR and the direction of the effect

# Consider that OR = exp(coeff.)
# Consider also, that it holds: exp(0)=1
# Therefore, OR = exp(coeff.) < 1 when coeff.<0 (hence a negative coefficient)
# Likewise, OR = exp(coeff.) > 1 when coeff.>0 (hence a positive coefficient)

# To summarize, an odds ratio that is larger than 1 indicates a positive effect of the variable
#   and an odds ratio that is smaller than 1 indicates a negative effect of the variable



#%% Task 4 - Calculating and interpreting the marginal effects

#__ Calculating and displaying marginal effects
marginal_ame_1 = results_1.get_margeff(at="overall") # AME
print(marginal_ame_1.summary())

marginal_mem_1 = results_1.get_margeff(at="mean") # MEM
print(marginal_mem_1.summary())


# AME for age
#   On average, a one-year increase in age increases the probability to vote 
#   by 0.24 percentage points
# AME for education = 2
#   On average, a person with high education is 15.0 percentage points more 
#   likely to vote than a person with no or low education
# AME for education = 1
#   On average, a person with medium education is 6.4 percentage points more 
#   likely to vote than a person with no or low education



#%%  Task 5 -  logistic regression with inclusion of the interest in politics

# 5a. Repeating the logistic regression
#__ Describing the model
model_2 = smf.logit("""voted ~ male + age + C(education) + C(media, Treatment(reference=4)) + 
                    C(political_interest, Treatment(reference=5))""", data = vote_clean)

#__ Fitting the model
results_2 = model_2.fit()

#__ Summarizing the model
# Printing a comprehensive overview
print(results_2.summary())

# The pseudo r square increases from 0.078 to 0.143, so there seems to be a notable improvement of the model



# 5b. Effect of the interest in politics on the voting behavior

# A high value of the media variable implies low monitoring of political developments in the media
# # A high value of the interest variable implies low interest of political developments in the media

# Calculating odds ratios
print(np.exp(results_2.params))
# Calculating AMEs
marginal_ame_2 = results_2.get_margeff(at="overall") # AME
print(marginal_ame_2.summary())

# To interpret the effect of the interest in politics we can either look at the 
#   log odds, the odds ratios or the AMEs, the effect remains the same.
#   The results overall show, that an increase in political interest and an increase in monitoring
#   politics in the media both are associated with an increased probability to vote.
#   Further, the probability increases monotonously with the level of interest and media monitoring.



# 5c. Comparing the AIC values of the two regression models

# Displaying the AIC value (Akaike Information Criterion)
print("The AIC of the first model is: " + str(results_1.aic))
print("The AIC of the second model is: " + str(results_2.aic))

# The second model has the lower AIC, it is therefore to be preferred over the first model.


#%%