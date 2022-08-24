import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns

# path = r'mini_project/train.csv'
path = r'train.csv'
df = pd.read_csv(path, low_memory=False)

# df.info()
df.groupby([df.Customer_ID, df.Age])

def frequency(data, column_name):
    return data[column_name].groupby([data.Customer_ID]).describe()

df['Age'] = df['Age'].str.replace('_', '')
print('1')
df['Annual_Income'] = df['Annual_Income'].str.replace('_', '')
df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].astype('str')
df['Num_of_Loan'] = df['Num_of_Loan'].str.replace('_', '')
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].str.replace('_', '')
df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].str.replace('_', '0')
df['Amount_invested_monthly'] = df['Amount_invested_monthly'].str.replace('_', '')
# describe = df.describe()

print('2')
age_groupby = frequency(df, 'Age')
print('3')
SSN_groupby = frequency(df, 'SSN')
print('4')
Occupation_groupby = frequency(df, 'Occupation')
print('5')
Monthly_Inhand_Salary_groupby = frequency(df, 'Monthly_Inhand_Salary')
print('6')


for i in tqdm(range(len(df))):
    if len(df['Age'][i]) != 2:
        df['Age'][i] = age_groupby['top'].loc[df['Customer_ID'][i]]
    else:
        continue

for i in tqdm(range(len(df))):
    if list(df['SSN'][i])[0].isnumeric() is False:
        df['SSN'][i] = SSN_groupby['top'].loc[df['Customer_ID'][i]]
    else:
        continue

for i in tqdm(range(len(df))):
    if list(df['Occupation'][i])[0].isalpha() is False:
        df['Occupation'][i] = Occupation_groupby['top'].loc[df['Customer_ID'][i]]
    else:
        continue

for i in tqdm(range(len(df))):
    if list(df['Monthly_Inhand_Salary'][i])[0].isnumeric() is False:
        df['Monthly_Inhand_Salary'][i] = Monthly_Inhand_Salary_groupby['top'].loc[df['Customer_ID'][i]]
    else:
        continue


df['Age'] = df['Age'].astype('float32')
df['Annual_Income'] = df['Annual_Income'].astype('float32')
df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].astype('float32')
df['Num_of_Loan'] = df['Num_of_Loan'].astype('float32')
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].astype('float32')
df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].astype('float32')
df['Amount_invested_monthly'] = df['Amount_invested_monthly'].astype('float32')
describe = df.describe()
print('7')
df.columns
df.info()

