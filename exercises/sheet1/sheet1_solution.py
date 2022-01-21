# -*- coding: utf-8 -*-
"""
Exercise sheet 1 - Solution

@author: Julia Jerke

Date: 2021-10-22
"""

#%%

import pandas as pd
import numpy as np
from pathlib import Path


#%% Task 1 - Loading the data set

path = Path(__file__).parent / 'dogs_of_zurich.csv'
dogs = pd.read_csv(path)



#%% Task 2 - How many dogs?

dogs.info()
# There are currently 8574 dogs registered in Zurich.



#%% Task 3 - Missing values

dogs.info()
# The variables with a high number of missings are: RASSE1_MISCHLING, RASSE2 and RASSE2_MISCHLING

# Dropping the columns from the data frame
dogs.drop(columns=["RASSE1_MISCHLING","RASSE2","RASSE2_MISCHLING"], inplace=True)

# Number of cases with missing values
dogs.dropna().info()
# There are 8574 - 8565 = 9 rows with missing values



#%% Task 4 - Renaming of the variables
dogs.rename(str.lower, axis=1, inplace=True)
dogs.rename({"halter_id": "owner_id", "alter": "age", "geschlecht": "sex",
             "stadtkreis": "district", "stadtquartier": "quarter",
             "rasse1": "breed", "rassentyp": "type",
             "geburtsjahr_hund": "year_dog", "geschlecht_hund": "sex_dog",
             "hundefarbe": "color"},
            inplace=True,axis=1, errors="raise")



#%% Task 5 - Dogs per owner

# Creating a series that contains the owner ID and the number of dogs per owner
owners = pd.DataFrame({"number_of_dogs": dogs.owner_id.value_counts()})

# Setting the index of the owner data frame to "owner_id"
owners.index.name = "owner_id"


owners.shape
owner_number = owners.shape[0] # we can slice the value of interest by using brackets and the position
print(owner_number)
# There are currently 7862 dog owners in Zurich


one_dog = owners.value_counts(sort=False)[1]
more_than_one = owner_number - one_dog
print(more_than_one)
# 592 people own more than one dog
owners.value_counts()
# The highest number of dogs a single person owns is 13


print(owners[owners.number_of_dogs==13])
# The owner with the ID 105585 has 13 dogs

dogs[dogs.owner_id==105585].breed.value_counts()
# This person has 5 Chinese Crested, 5 Papillon, 2 Chihuahua and 1 Zwergspitz



#%% Task 6 - Age of dogs

dogs["age_dog"] = 2021 - dogs.year_dog

dogs.age_dog.describe()
# The dogs are on average 7.27 years old and the oldest dog seems to be 2010 years old.

# Alternatively, call the mean and max value explicitely
dogs.age_dog.mean()
dogs.age_dog.max()


print(dogs.sort_values(by="age_dog").age_dog)


# The oldest dog worldwide died with 30 years and no dog can have a negative age
dogs.loc[dogs.age_dog < 0,"age_dog"] = np.nan
dogs.loc[dogs.age_dog > 30,"age_dog"] = np.nan


# The dogs are on average 6.9 years old and the oldest dog seems to be 29 years old
dogs.age_dog.mean()
dogs.age_dog.max()



#%% Task 7 - Dog breeds and districts

dogs.breed.value_counts().head(5)
# The most favorite breeds are: Mischling klein, Chihuahua, Mischling gross, Labrador Retriever, Französische Bulldogge


dogs.district.value_counts().head(1)
# The most dogs live in district 11


district_ages = dogs.groupby("district").age_dog.mean()
oldest_dogs = district_ages.sort_values(ascending=False).head(1)
print(oldest_dogs)
# The oldest dogs live in district 10


dogs[dogs.breed=="Dachshund"].district.value_counts().head(1)
# The most wiener dogs live in district 7


dogs[dogs.district==7].breed.value_counts().head(1)
# The favorite dog in district 7 is the Labrador retriever



#%% Task 8 - Owner characteristics

# Important for that task:
# We cannot use the distribution of age and gender in the raw data set.
# Some owners appear more than once because they own more than one dog.
# The results would therefore be biased.
# We need to aggregate for the different owners


# Recoding the information on the owner sex into a new numeric variable
dogs["sex_int"] = 0 # group male
dogs.loc[dogs.sex=="w","sex_int"] = 1
pd.crosstab(dogs.sex,dogs.sex_int) # check the recoding


# Recoding the information on the owner age into a new numeric variable
dogs["age_int"] = 1 # group 11-20
dogs.loc[dogs.age=="21-30","age_int"] = 2
dogs.loc[dogs.age=="31-40","age_int"] = 3
dogs.loc[dogs.age=="41-50","age_int"] = 4
dogs.loc[dogs.age=="51-60","age_int"] = 5
dogs.loc[dogs.age=="61-70","age_int"] = 6
dogs.loc[dogs.age=="71-80","age_int"] = 7
dogs.loc[dogs.age=="81-90","age_int"] = 8
dogs.loc[dogs.age=="91-100","age_int"] = 9
pd.crosstab(dogs.age,dogs.age_int) # check the recoding


# Creating a temporary data frame that contains the aggregated information on owner sex and age
owner_temp = pd.DataFrame({"sex": dogs.groupby("owner_id").sex_int.mean(),
                          "age": dogs.groupby("owner_id").age_int.mean()})
owners.set_index("owner_id",inplace=True)


# Adding the two new owner variales to the already existing owner data frame and deleting the temporary data frame
owners = owners.join(owner_temp, on="owner_id")
del owner_temp


# Getting the relevant information
owners.sex.value_counts()
# There 5379 female and 2483 male dog owners
owners.age.value_counts()
# E.g.: Most dog owners are between 31 and 40 years old, the overall range is 11 to 100 years



#%% Task 9 - Exporting the data frame

dogs.to_excel("dogs_of_zurich.xlsx")
