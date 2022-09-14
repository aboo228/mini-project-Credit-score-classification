import pandas as pd
import numpy as np
import seaborn as sns
from sqlalchemy.orm.loading import instances
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import os


train_path = r'train.csv'
test_path = r'test.csv'
train_df = pd.read_csv(train_path,low_memory=False)
test_df = pd.read_csv(test_path)
train_targets=train_df.iloc[:,-1]
train_df=train_df.iloc[:,:-1]
train_df.info()
train_df.isna().sum()
# numeric columns dtype is object, so we need to convert it to integer type
numeric_mixedtype_columns = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                             'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance']
# todo: Credit_History_Age: convert str to num of month

'''remove symbols from age column values '''

for i in tqdm(range(len(numeric_mixedtype_columns))):
    train_df.loc[:, numeric_mixedtype_columns[i]] = train_df.loc[:, numeric_mixedtype_columns[i]].astype('str')
    train_df.loc[:, numeric_mixedtype_columns[i]] = (
            train_df.loc[:, numeric_mixedtype_columns[i]].str.replace('_', '')).str.replace('-', '')
    train_df.loc[:, numeric_mixedtype_columns[i]] = train_df.loc[:, numeric_mixedtype_columns[i]].replace('', None)


train_df.loc[:, numeric_mixedtype_columns[0]] = train_df.loc[:, numeric_mixedtype_columns[0]].astype('int32')


# train_df.loc[:, numeric_mixedtype_columns[0]].describe()

'''check valid ages'''
'''min 14 check and its valid'''
print(train_df[train_df.loc[:, 'Age'] == 95])
invalid_age_instances = train_df.index[train_df.loc[:, 'Age'] >= 95].to_list()
invalid_age_custid = train_df.loc[invalid_age_instances, 'Customer_ID'].to_list()
invalid_delayed_pay_ins = train_df.index[train_df.loc[:, 'Num_of_Delayed_Payment'].isnull()].to_list()
invalid_delayed_pay_custid = train_df.loc[invalid_delayed_pay_ins, 'Customer_ID'].to_list()
train_df.loc[:, 'Num_of_Delayed_Payment'].fillna('-1000', inplace=True)

invalid_values_instances = invalid_age_instances, invalid_delayed_pay_ins
invalid_index_by_custid = invalid_age_custid, invalid_delayed_pay_custid
#
roll_columns = ['Age', 'Num_of_Delayed_Payment']
for column in tqdm(range(len(roll_columns))):
    for instance, id in zip(invalid_values_instances[column], invalid_index_by_custid[column]):
        fill = None
        if 0 < instance % 8 < 7:
            fill = np.max((train_df.loc[:, roll_columns[column]][train_df.loc[:, 'Customer_ID'] == id]).loc[
                              [instance - 1, instance + 1]].astype('int32'))
            if len(str(fill)) <= 2:
                train_df.loc[instance, roll_columns[column]] = fill
            else:
                train_df.loc[instance, roll_columns[column]] = (
                    train_df.drop(invalid_values_instances[column]).loc[:, roll_columns[column]][train_df.loc[:, 'Customer_ID'] == id]).value_counts().idxmax()
        else:
            train_df.loc[instance, roll_columns[column]] = (
                train_df.drop(invalid_values_instances[column]).loc[:, roll_columns[column]][train_df.loc[:, 'Customer_ID'] == id]).value_counts().idxmax()

train_df.loc[:, 'Age'] = train_df.loc[:, 'Age'].astype('int32')

'''occupation and name columns complete missing values'''

unique_occupation = train_df.loc[:, 'Occupation'].unique()
invalid_occupation = unique_occupation[1]
del unique_occupation

unique_customer_ids = train_df.loc[:, 'Customer_ID'].unique()
unique_name = train_df.loc[:, 'Name'].unique()
per_customer = None
columns_to = ['Name', 'Monthly_Inhand_Salary', 'Num_Credit_Inquiries', 'Occupation', 'Credit_Mix', 'Num_Bank_Accounts',
              'Num_Credit_Card', 'Num_of_Loan']
'''target invalid values'''

# unknown_values_indexes = [[[train_df.index[train_df.loc[:, columns_to[column]] == 'Unknown']] for column in\
# range(len(columns_to))]]

'''check num of banks accounts >11'''
'''check num of credit card>11'''
'''check num of loan >9'''

criterions = [invalid_occupation, '_', 11, 11, 9]
'''convert to None soo that the will be no value'''
[train_df.loc[:, columns_to[i+3]].replace(criterions[i], None, inplace=True) for i in range(2)]

unknown_values_indexes = []
for col_i in range(len(columns_to)):
    if col_i <= 4:
        unknown_values_indexes.append(train_df.index[train_df.loc[:, columns_to[col_i]].isnull()])
    # elif 2 < int(col_i) <= 4:
    #     unknown_values_indexes.append(train_df.index[train_df.loc[:, columns_to[col_i]] == criterions[col_i - 3]])
    else:
        unknown_values_indexes.append(
            train_df.index[train_df.loc[:, columns_to[col_i]].astype('int32') > criterions[col_i - 3]])

for column in tqdm(range(len(columns_to))):
    for unknown_val_index in unknown_values_indexes[column]:
        customer_id = train_df.loc[unknown_val_index, 'Customer_ID']
        start_check_index = 0
        if unknown_val_index < 8:
            start_check_index = 0

        else:
            start_check_index = unknown_val_index - 8


        train_df.loc[unknown_val_index, columns_to[column]] = (
            train_df.loc[start_check_index:unknown_val_index + 8, columns_to[column]][
                train_df.loc[start_check_index:unknown_val_index + 8, 'Customer_ID'] == customer_id]).drop(
            unknown_val_index).value_counts().idxmax()

'''convert credit history age to month unit'''
instance_to_convert = train_df.index[~pd.isnull(train_df.loc[:, 'Credit_History_Age'])]
instance_to_fill = train_df.index[pd.isnull(train_df.loc[:, 'Credit_History_Age'])]
_ = train_df.loc[instance_to_convert, 'Credit_History_Age'].str.split(' ')
train_df.loc[instance_to_convert, 'Credit_History_Age'] = (_.str.get(0)).astype('int32') * 12 + (_.str.get(3)).astype(
    'int32')
#
'''fill mising values in credit age feature'''
inst_nan_credithist=train_df.index[train_df.Credit_History_Age.isnull()]
credit_age_values=train_df.loc[:,['Credit_History_Age','Customer_ID']].drop(inst_nan_credithist,axis=0)
customerid=None
index_min_age=None
# todo: improve o()
for i in tqdm(inst_nan_credithist):
    customerid=train_df.loc[i, 'Customer_ID']
    index_min_age = credit_age_values[credit_age_values.Customer_ID==customerid].sort_values('Credit_History_Age').index[0]
    train_df.loc[i, 'Credit_History_Age']= (i-index_min_age)+credit_age_values.loc[index_min_age,'Credit_History_Age']

'''extricate loans types to convert loans types to columns '''
unique_loans_types = []
[unique_loans_types.extend(train_df.loc[:, 'Type_of_Loan'].str.split(',').str.get(i).str.strip().unique().tolist()) for
 i in range(9)]
unique_loans_types = list(set(unique_loans_types))
'''convert object to columns'''
_ = np.zeros((train_df.shape[0], len(unique_loans_types)))
get_dummies = pd.DataFrame(_, columns=unique_loans_types)
col = []
for i in tqdm(range(9)):
    col = train_df.loc[:, 'Type_of_Loan'].str.split(',').str.get(i).str.strip().unique().tolist()
    get_dummies.loc[:, col] = get_dummies.loc[:, col] + pd.get_dummies(
        train_df.loc[:, 'Type_of_Loan'].str.split(',').str.get(i).str.strip())
train_df.drop('Type_of_Loan', inplace=True, axis=1)
train_df.rename(columns={'Credit_History_Age': 'Credit_Months_History_Age'})
train_df = pd.concat([train_df, get_dummies], axis=1)

train_df.drop(train_df.iloc[:, 26], axis=1, inplace=True)
train_df.iloc[:, :] = train_df.iloc[:, :].replace('nan', None)
train_df.iloc[:, :] = train_df.iloc[:, :].replace('None', None)
columns_with_null = train_df.columns[train_df.isnull().sum() > 0]
# for more accuracy need to be check each customer std, we will omit that and just fill missing data with the mean of the customer
for column in tqdm(columns_with_null[:-1]):
    insta=train_df.index[train_df[column].isnull()]
    customer_na_id=train_df.loc[insta,'Customer_ID'].to_list()
    for indicator,id in zip(insta,customer_na_id):
        _=train_df.loc[:,column][train_df.loc[:,'Customer_ID']==id]
        if column=='Num_of_Delayed_Payment':
            train_df.loc[indicator, column]=int(np.round(_.astype(np.float32).mean()))
        else:
            train_df.loc[indicator, column]=np.round(_.astype(np.float32).mean(),2)


'''drop not important columns' we can do feature engineering on name column by classification by sex '''
train_df.drop(['ID', 'Name', 'SSN'], axis=1, inplace=True)
train_df.drop(['Customer_ID', 'Month', 'Payment_Behaviour'], axis=1, inplace=True)


get_dum_col = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount']
dumdum = pd.get_dummies(train_df.loc[:, get_dum_col])
train_df = pd.concat([train_df, dumdum], axis=1)
train_df.drop(train_df.loc[:, get_dum_col], axis=1, inplace=True)



# todo: improve dtype convert

train_df.iloc[:, 15] = train_df.iloc[:, 15].astype('str').replace('10000', None)
train_df.iloc[:, 15] = train_df.iloc[:, 15].replace('None', None)
train_df.iloc[:, 15] = train_df.iloc[:, 15].astype('float32')
train_df.iloc[:, 16] = train_df.iloc[:, 16].astype('float32')
train_df.iloc[train_df.index[train_df.iloc[:, 16].astype('float32') > 5000000000], 16] = None
train_df.iloc[:, :] = train_df.iloc[:, :].replace('nan', None)
train_df.iloc[:, 17:35] = train_df.iloc[:, 17:35].astype('int32')
'''predict missing values'''


instances_with_null = train_df.index[train_df.isnull().sum(axis=1) > 0]
columns_with_null = train_df.columns[train_df.isnull().sum() > 0]
instances_to_predict = train_df.iloc[instances_with_null, :]
train_df=pd.concat([train_df, train_targets],axis=1)

# filling more missing data
# for column in columns_with_null[:-1]:
#     insta=train_df.index[train_df[column].isnull()]
#     customer_na_id=train_df['Customer_Id'].
# export as csv_file
train_df.to_csv('train_df.csv', index=False)
