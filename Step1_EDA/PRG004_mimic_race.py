# load library
import numpy as np
import pandas as pd
import wfdb
import joblib
import glob
import tqdm
from IPython.display import display
from sklearn.model_selection import train_test_split

merge_df_sel2 = joblib.load("dataset/df/merge_df_sel2")


merge_df_sel2.race.value_counts()

# https://github.com/rmovva/granular-race-disparities_MLHC23/blob/main/analysis/race_categories.py

# We exclude 'OTHER' from the analysis in the paper.
coarse_races = [
    'WHITE',
    'BLACK/AFRICAN AMERICAN',
    'HISPANIC OR LATINO',
    'ASIAN',
#     'OTHER',
]

coarse_abbrev = {
    'WHITE': 'White',
    'BLACK/AFRICAN AMERICAN': 'Black',
    'HISPANIC OR LATINO': 'Hispanic/Latino',
    'ASIAN': 'Asian',
    'AVERAGE': 'Average', # this is for cases where we want to average over coarse groups
}

granular_to_coarse = {
    'HISPANIC OR LATINO': 'HISPANIC OR LATINO',
    'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC OR LATINO', 
    'HISPANIC/LATINO - DOMINICAN': 'HISPANIC OR LATINO', 
    'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC OR LATINO', 
    'HISPANIC/LATINO - SALVADORAN': 'HISPANIC OR LATINO', 
    'HISPANIC/LATINO - MEXICAN': 'HISPANIC OR LATINO', 
    'HISPANIC/LATINO - COLUMBIAN': 'HISPANIC OR LATINO', 
    'HISPANIC/LATINO - HONDURAN': 'HISPANIC OR LATINO', 
    'HISPANIC/LATINO - CUBAN': 'HISPANIC OR LATINO',
    'HISPANIC/LATINO - CENTRAL AMERICAN': 'HISPANIC OR LATINO',
    'SOUTH AMERICAN': 'HISPANIC OR LATINO',
    
    'ASIAN': 'ASIAN',
    'ASIAN - CHINESE': 'ASIAN',
    'ASIAN - SOUTH EAST ASIAN': 'ASIAN',
    'ASIAN - ASIAN INDIAN': 'ASIAN',
    'ASIAN - KOREAN': 'ASIAN',
    
    'WHITE': 'WHITE',
    'WHITE - OTHER EUROPEAN': 'WHITE',
    'WHITE - RUSSIAN': 'WHITE',
    'WHITE - EASTERN EUROPEAN': 'WHITE',
    'WHITE - BRAZILIAN': 'WHITE',
    'PORTUGUESE': 'WHITE',
    
    'BLACK/AFRICAN AMERICAN': 'BLACK/AFRICAN AMERICAN',
    'BLACK/CAPE VERDEAN': 'BLACK/AFRICAN AMERICAN',
    'BLACK/CARIBBEAN ISLAND': 'BLACK/AFRICAN AMERICAN',
    'BLACK/AFRICAN': 'BLACK/AFRICAN AMERICAN',
    
#     'AMERICAN INDIAN/ALASKA NATIVE': 'OTHER',
#     'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'OTHER',
#     'MULTIPLE RACE/ETHNICITY': 'OTHER',
#     'UNKNOWN': 'OTHER',
#     'PATIENT DECLINED TO ANSWER': 'OTHER',
#     'UNABLE TO OBTAIN': 'OTHER',
}

granular_abbrev = {
    'HISPANIC OR LATINO': 'HISPANIC OR LATINO*',
    'HISPANIC/LATINO - PUERTO RICAN': 'PUERTO RICAN', 
    'HISPANIC/LATINO - DOMINICAN': 'DOMINICAN', 
    'HISPANIC/LATINO - GUATEMALAN': 'GUATEMALAN', 
    'HISPANIC/LATINO - SALVADORAN': 'SALVADORAN', 
    'HISPANIC/LATINO - MEXICAN': 'MEXICAN', 
    'HISPANIC/LATINO - COLUMBIAN': 'COLOMBIAN', 
    'HISPANIC/LATINO - HONDURAN': 'HONDURAN', 
    'HISPANIC/LATINO - CUBAN': 'CUBAN',
    'HISPANIC/LATINO - CENTRAL AMERICAN': 'CENTRAL AMERICAN',
    'SOUTH AMERICAN': 'SOUTH AMERICAN',
    
    'ASIAN': 'ASIAN*',
    'ASIAN - CHINESE': 'CHINESE',
    'ASIAN - SOUTH EAST ASIAN': 'SE ASIAN',
    'ASIAN - ASIAN INDIAN': 'INDIAN',
    'ASIAN - KOREAN': 'KOREAN',
    
    'WHITE': 'WHITE*',
    'WHITE - OTHER EUROPEAN': 'OTHER EUR',
    'WHITE - RUSSIAN': 'RUSSIAN',
    'WHITE - EASTERN EUROPEAN': 'EASTERN EUR',
    'WHITE - BRAZILIAN': 'BRAZILIAN',
    'PORTUGUESE': 'PORTUGUESE',
    
    'BLACK/AFRICAN AMERICAN': 'BLACK*', 
    'BLACK/CAPE VERDEAN': 'CAPE VERDEAN',
    'BLACK/CARIBBEAN ISLAND': 'CARIBBEAN',
    'BLACK/AFRICAN': 'AFRICAN',
    
#     'AMERICAN INDIAN/ALASKA NATIVE': 'AMERICAN INDIAN',
#     'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'PACIFIC ISLANDER',
#     'MULTIPLE RACE/ETHNICITY': 'MULTIRACIAL',
#     'UNKNOWN': 'UNKNOWN',
#     'PATIENT DECLINED TO ANSWER': 'DECLINE TO ANSWER',
#     'UNABLE TO OBTAIN': 'UNABLE TO OBTAIN',
}

coarse_to_granular = {
    'WHITE': [
        'WHITE',
        'WHITE - OTHER EUROPEAN',
        'WHITE - RUSSIAN',
        'WHITE - EASTERN EUROPEAN',
        'WHITE - BRAZILIAN',
        'PORTUGUESE',
    ],
    'BLACK/AFRICAN AMERICAN': [
        'BLACK/AFRICAN AMERICAN',
        'BLACK/CAPE VERDEAN',
        'BLACK/CARIBBEAN ISLAND',
        'BLACK/AFRICAN',
    ],
    'HISPANIC OR LATINO': [
        'HISPANIC OR LATINO',
        'HISPANIC/LATINO - PUERTO RICAN', 
        'HISPANIC/LATINO - DOMINICAN', 
        'HISPANIC/LATINO - GUATEMALAN', 
        'HISPANIC/LATINO - SALVADORAN', 
        'HISPANIC/LATINO - MEXICAN', 
        'HISPANIC/LATINO - COLUMBIAN', 
        'HISPANIC/LATINO - HONDURAN', 
        'HISPANIC/LATINO - CUBAN',
        'HISPANIC/LATINO - CENTRAL AMERICAN',
        'SOUTH AMERICAN'
    ],
    'ASIAN': [
        'ASIAN',
        'ASIAN - CHINESE', 
        'ASIAN - SOUTH EAST ASIAN', 
        'ASIAN - ASIAN INDIAN', 
        'ASIAN - KOREAN',
    ],
#     'OTHER': [
#         'AMERICAN INDIAN/ALASKA NATIVE',
#         'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
#         'MULTIPLE RACE/ETHNICITY',
#     ]
}


merge_df_sel2['race2'] = merge_df_sel2.race.replace(granular_to_coarse).replace(coarse_abbrev)


joblib.dump(merge_df_sel2,"dataset/df/merge_df_sel3")

# merge_df_sel2['race2'].value_counts()
# race2
# White                                        330563
# Black                                         74700
# Hispanic/Latino                               25520
# UNKNOWN                                       16707
# OTHER                                         14217
# Asian                                         13627
# UNABLE TO OBTAIN                               2712
# PATIENT DECLINED TO ANSWER                     1685
# AMERICAN INDIAN/ALASKA NATIVE                   964
# MULTIPLE RACE/ETHNICITY                         503
# NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER       370