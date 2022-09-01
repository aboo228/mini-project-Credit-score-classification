import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
# path = r'mini_project/train.csv'
train_path = r'train.csv'

# train_path = r'C:\Users\sagir\Documents\database\credit_score_classification\train.csv'
# test_path = r'C:\Users\sagir\Documents\database\credit_score_classification\test.csv'
train_df = pd.read_csv(train_path)
# test_df = pd.read_csv(test_path)
train_df.info()
train_df.isna().sum()
# numeric columns dtype is object, so we need to convert it to integer type
numeric_mixedtype_columns = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                             'Outstanding_Debt', 'Amount_invested_monthly','Monthly_Balance']
# todo: Credit_History_Age: convert str to num of month

'''remove symbols from age column values '''
# train_df.loc[:,'Age']=(train_df.loc[:,'Age'].str.replace('_','')).str.replace('-','')

for i in tqdm(range(len(numeric_mixedtype_columns))):
    train_df.loc[:, numeric_mixedtype_columns[i]] = train_df.loc[:, numeric_mixedtype_columns[i]].astype('str')
    train_df.loc[:, numeric_mixedtype_columns[i]] = (
            train_df.loc[:, numeric_mixedtype_columns[i]].str.replace('_', '')).str.replace('-', '')
    train_df.loc[:, numeric_mixedtype_columns[i]] = train_df.loc[:, numeric_mixedtype_columns[i]].replace('', None)


train_df.loc[:, numeric_mixedtype_columns[0]] = train_df.loc[:, numeric_mixedtype_columns[0]].astype('int32')
# train_df.loc[:, numeric_mixedtype_columns[3]] = train_df.loc[:, numeric_mixedtype_columns[3]].astype('int32')

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
            if len(str(fill)) <= 3:
                train_df.loc[instance, roll_columns[column]] = fill
            else:
                train_df.loc[instance, roll_columns[column]] = (
                    train_df.loc[:, roll_columns[column]][train_df.loc[:, 'Customer_ID'] == id]).drop(
                    instance).value_counts().idxmax()
        else:
            train_df.loc[instance, roll_columns[column]] = (
                train_df.loc[:, roll_columns[column]][train_df.loc[:, 'Customer_ID'] == id]).drop(
                instance).value_counts().idxmax()

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
# unknown_values_indexs = [[[train_df.index[train_df.loc[:, columns_to[column]] == 'Unknown']] for column in range(len(columns_to))]]

'''check num of banks accounts >11'''
'''check num of credit card>11'''
'''check num of loan >9'''

criterions = [invalid_occupation, '_', 11, 11, 9]
'''covert to None soo that the will be no value'''
[train_df.loc[:,columns_to[i+3]].replace(criterions[i],None,inplace=True) for i in range(2)]

unknown_values_indexs = []
for col_i in range(len(columns_to)):
    if col_i <= 4:
        unknown_values_indexs.append(train_df.index[train_df.loc[:, columns_to[col_i]].isnull()])
    # elif 2 < int(col_i) <= 4:
    #     unknown_values_indexs.append(train_df.index[train_df.loc[:, columns_to[col_i]] == criterions[col_i - 3]])
    else:
        unknown_values_indexs.append(
            train_df.index[train_df.loc[:, columns_to[col_i]].astype('int32') > criterions[col_i - 3]])

for column in tqdm(range(len(columns_to))):
    for unknown_val_index in unknown_values_indexs[column]:
        customer_id = train_df.loc[unknown_val_index, 'Customer_ID']
        start_check_index = 0
        if unknown_val_index < 8:
            start_check_index = 0

        else:
            start_check_index = unknown_val_index - 8

        # customer_id_index=train_df.index[train_df.loc[]]
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

# for i in tqdm(range(len(columns_to))):
#     for unknown_val_index in (unknown_values_indexs[i]):
#         customer_id = train_df.loc[unknown_val_index, 'Customer_ID']
#         train_df.loc[unknown_val_index, columns_to[i]] = (
#         train_df.loc[:, columns_to[i]][train_df.loc[:, 'Customer_ID'] == customer_id]).drop(
#             unknown_val_index).value_counts().idxmax()

# unique_customer_ids = train_df.loc[:, 'Customer_ID'].unique()
# for id in tqdm(unique_customer_ids):
#     per_customer=train_df.loc[:,columns_to][train_df.loc[:,'Customer_ID']==id]
#     invalid_value=None
#     invalid_index=None
#     for i in range(len(per_customer.columns)):
#         if per_customer.iloc[:,i].value_counts().max()<8:
#             invalid_value=per_customer.iloc[:, i].value_counts().idxmin()
#             invalid_index=per_customer.index[per_customer.iloc[:, i]==invalid_value]
#             for index in invalid_index:
#                 train_df.loc[index,columns_to[i]]=per_customer.iloc[:, i].value_counts().idxmax()
#


# for id in tqdm(unique_customer_ids):
#     per_customer=train_df.loc[:,columns_to][train_df.loc[:,'Customer_ID']==id]
#     invalid_value=None
#     invalid_index=None
#     for i in range(len(per_customer.columns)):
#         if per_customer.iloc[:,i].value_counts().max()<8:
#             invalid_value=per_customer.iloc[:, i].value_counts().idxmin()
#             invalid_index=per_customer.index[per_customer.iloc[:, i]==invalid_value]
#             for index in invalid_index:
#                 train_df.loc[index,columns_to[i]]=per_customer.iloc[:, i].value_counts().idxmax()
#


# invalid_occupation=unique_occupation[1]
#
#
# for instance,age in tqdm(enumerate(train_df.loc[:, 'Age'])):
#     if len(str(age)) > 2:
#         cus_id = train_df.loc[instance, 'Customer_ID']
#         if 0 <instance % 8 < 7:
#
#             fill=np.max((train_df.loc[:,'Age'][train_df.loc[:,'Customer_ID']==cus_id]).loc[[instance-1,instance+1]])
#             if len(str(fill))==2:
#                 train_df.loc[instance, 'Age']=fill
#         else:
#             train_df.loc[instance, 'Age']=round(np.mean((train_df.loc[:, 'Age'][train_df.loc[:, 'Customer_ID']==cus_id]).drop(instance),axis=0))
# train_df.loc[:,'Age']=train_df.loc[:,'Age'].astype('int32')
# train_df.loc[:, numeric_mixedtype_columns] = train_df.loc[:, numeric_mixedtype_columns].astype('int32')

train_df.drop(train_df.iloc[:, 27], axis=1, inplace=True)
col = [7, 8,15, 19, 22, 23, 25, 26]
col_num_check = train_df.iloc[:, col].copy()
col_to_float=['Outstanding_Debt','Credit_History_Age','Amount_invested_monthly','Monthly_Balance','Annual_Income','Changed_Credit_Limit']
col_to_int=['Num_of_Loan', 'Num_of_Delayed_Payment']
#
# for i in range(len(yu)):
#     yu[i]=pd.to_numeric(col_num_check.loc[:,yu[i]])

'''drop not important columns' we can do feature engineering on name column by classification by sex '''
train_df.drop(['ID','Name','SSN'],axis=1, inplace=True)
train_df.drop(['Customer_ID','Month','Payment_Behaviour'],axis=1, inplace=True)

for i in range(len(col_to_float)):
    train_df.loc[:,col_to_float[i]]=train_df.loc[:,col_to_float[i]].astype('float64')
    train_df.loc[:,col_to_float[i]]=pd.to_numeric(train_df.loc[:,col_to_float[i]])

for i in range(len(col_to_int)):
    train_df.loc[:,col_to_int[i]]=train_df.loc[:,col_to_int[i]].astype('int64')
    train_df.loc[:,col_to_int[i]] =pd.to_numeric(train_df.loc[:,col_to_int[i]])

# col_num_check.loc[:,'Changed_Credit_Limit']=pd.to_numeric(col_num_check.loc[:,'Changed_Credit_Limit'])
# col_num_check.loc[:,'Monthly_Balance']=pd.to_numeric(col_num_check.loc[:,'Monthly_Balance'])
# col_num_check.dropna(axis=0, inplace=True)
# # col_num_check.fillna(0,inplace=True)
# for i in range(col_num_check.shape[1] - 1):
#     col_num_check.iloc[:, i] = col_num_check.iloc[:, i].astype('float64')
pd.get_dummies(train_df)