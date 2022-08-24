import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns

# path = r'mini_project/train.csv'
path = r'train.csv'
df = pd.read_csv(path, low_memory=False)

describe = df.describe()
# df.info()
df.groupby([df.Customer_ID, df.Age])

def frequency(data, column_name):
    return data[column_name].groupby([data.Customer_ID]).describe()

df['Age'] = df['Age'].str.replace('_', '')
print('1')
df['Annual_Income'] = df['Annual_Income'].str.replace('_', '')
df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].astype('str')
df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].str.replace('_', '')
# print('2')
# age_groupby = df['Age'].groupby([df.Customer_ID]).describe()
# print('3')
# SSN_groupby = df['SSN'].groupby([df.Customer_ID]).describe()
# print('4')
# Occupation_groupby = df['Occupation'].groupby([df.Customer_ID]).describe()
print('5')
# Monthly_Inhand_Salary_groupby = df['Monthly_Inhand_Salary'].groupby([df.Customer_ID]).describe()
Monthly_Inhand_Salary_groupby = frequency(df, 'Monthly_Inhand_Salary')
print('6')





for i in tqdm(range(len(df))):
    # if len(df['Age'][i]) == 2:
    #     continue
    # else:
    #     df['Age'][i] = age_groupby['top'].loc[df['Customer_ID'][i]]

    # print(list(df['SSN'][i])[0])
    # if list(df['SSN'][i])[0].isnumeric() is False:
    #     df['SSN'][i] = SSN_groupby['top'].loc[df['Customer_ID'][i]]
    # else:
    #     continue

    # print(list(df['Occupation'][i])[0])
    # if list(df['Occupation'][i])[0].isalpha() is False:
    #     df['Occupation'][i] = Occupation_groupby['top'].loc[df['Customer_ID'][i]]
    # else:
    #     continue

    # print(list(df['Monthly_Inhand_Salary'][i])[0])
    if list(df['Monthly_Inhand_Salary'][i])[0].isnumeric() is False:
        df['Monthly_Inhand_Salary'][i] = Monthly_Inhand_Salary_groupby['top'].loc[df['Customer_ID'][i]]
    else:
        continue


df['Age'] = df['Age'].astype('float32')
df['Annual_Income'] = df['Annual_Income'].astype('float32')
df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].astype('float32')

df.columns
# testdsd
#test shn------
asd