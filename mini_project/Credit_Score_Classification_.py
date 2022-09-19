import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import os

train_path = r'train.csv'
test_path = r'test.csv'
train_df = pd.read_csv(train_path, low_memory=False)
test_df = pd.read_csv(test_path)
train_targets = train_df.iloc[:, -1]
train_df = train_df.iloc[:, :-1]
train_df.info()
train_df.isna().sum()
# to see the null ratio for all instanses
print(train_df.isna().mean())
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
                    train_df.drop(invalid_values_instances[column]).loc[:, roll_columns[column]] \
                        [train_df.loc[:, 'Customer_ID'] == id]).value_counts().idxmax()
        else:
            train_df.loc[instance, roll_columns[column]] = (
                train_df.drop(invalid_values_instances[column]).loc[:, roll_columns[column]] \
                    [train_df.loc[:, 'Customer_ID'] == id]).value_counts().idxmax()

train_df.loc[:, ['Num_of_Loan', 'Age', 'Interest_Rate']] = train_df.loc[:,
                                                           ['Num_of_Loan', 'Age', 'Interest_Rate']].astype('int32')

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
[train_df.loc[:, columns_to[i + 3]].replace(criterions[i], None, inplace=True) for i in range(2)]

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
inst_nan_credithist = train_df.index[train_df.Credit_History_Age.isnull()]
credit_age_values = train_df.loc[:, ['Credit_History_Age', 'Customer_ID']].drop(inst_nan_credithist, axis=0)
customerid = None
index_min_age = None
# todo: improve o()
for i in tqdm(inst_nan_credithist):
    customerid = train_df.loc[i, 'Customer_ID']
    index_min_age = \
        credit_age_values[credit_age_values.Customer_ID == customerid].sort_values('Credit_History_Age').index[0]
    train_df.loc[i, 'Credit_History_Age'] = (i - index_min_age) + credit_age_values.loc[
        index_min_age, 'Credit_History_Age']

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
'''for more accuracy need to be check each customer std, 
    we will omit that and just fill missing data with the mean of the customer'''
for column in tqdm(columns_with_null[:-1]):
    insta = train_df.index[train_df[column].isnull()]
    customer_na_id = train_df.loc[insta, 'Customer_ID'].to_list()
    for indicator, id in zip(insta, customer_na_id):
        _ = train_df.loc[:, column][train_df.loc[:, 'Customer_ID'] == id]
        if column == 'Num_of_Delayed_Payment':
            train_df.loc[indicator, column] = int(np.round(_.astype(np.float32).mean()))
        else:
            train_df.loc[indicator, column] = np.round(_.astype(np.float32).mean(), 2)

'''drop not important columns' we can do feature engineering on name column by classification by sex '''
train_df.drop(['ID', 'Name', 'SSN'], axis=1, inplace=True)
train_df.drop(['Month', 'Payment_Behaviour'], axis=1, inplace=True)

get_dum_col = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount']
dumdum = pd.get_dummies(train_df.loc[:, get_dum_col])
train_df = pd.concat([train_df, dumdum], axis=1)
train_df.drop(train_df.loc[:, get_dum_col], axis=1, inplace=True)

# todo: improve dtype convert

train_df.iloc[:, 16] = train_df.iloc[:, 16].astype('str').replace('10000', None)
train_df.iloc[:, 16] = train_df.iloc[:, 16].replace('None', None)
train_df.iloc[:, 16] = train_df.iloc[:, 16].astype('float32')
train_df.iloc[:, 17] = train_df.iloc[:, 17].astype('float32')
train_df.iloc[train_df.index[train_df.iloc[:, 17].astype('float32') > 5000000000], 17] = None
train_df.iloc[:, :] = train_df.iloc[:, :].replace('nan', None)
train_df.iloc[:, 18:36] = train_df.iloc[:, 18:36].astype('int32')
train_df.iloc[:, 2] = train_df.iloc[:, 2].astype('float32')
col_to_float=train_df.dtypes.where(train_df.drop('Customer_ID',axis=1).dtypes=='O').dropna().index
train_df.loc[:,col_to_float] = train_df.loc[:,col_to_float].astype('float32')

'''predict missing values'''

# instances_with_null = train_df.index[train_df.isnull().sum(axis=1) > 0]
# columns_with_null = train_df.columns[train_df.isnull().sum() > 0]
# instances_to_predict = train_df.iloc[instances_with_null, :]
train_df = pd.concat([train_df, train_targets], axis=1)

# filling more missing data
# for column in columns_with_null[:-1]:
#     insta=train_df.index[train_df[column].isnull()]
#     customer_na_id=train_df['Customer_Id'].
# export as csv_file

# todo:check the missin values
'''check missing values in amount invested monthly column'''
# custidinvestnull=train_df.loc[:,'Customer_ID'][train_df[columns_with_null[2]].isna()].unique()
# investcustomerwithnull=train_df[train_df.loc[:,'Customer_ID'].isin(custidinvestnull)]

'''total emi per month invalid values, easy we can see that for example '''

'''check invalid values in interest rate column, the values not making sense, not for APR and not for montly rate,
i will assume that its annual interest rate(APR) for the sake of logical threshold (100) '''
intr_rate_invalid_values = train_df[train_df['Interest_Rate'] > 50].index
customerid_inter_rate = train_df.Customer_ID[intr_rate_invalid_values].unique()
# train_df.loc[:,['Interest_Rate','Customer_ID']].where(train_df.Customer_ID.isin(customerid_inter_rate)).dropna()

'''emi values that are outliers its seems that its invalid values, we can see the its not influencing other columns 
    also its high values for some customer that did not take loans at all,check each customer if the std for total emi is above 100,        
    we will check, and change all values the above 1000 '''
train_df.Total_EMI_per_month.where(train_df.Num_of_Loan != 0, 0, inplace=True)

customer_emi_outlier = train_df.groupby('Customer_ID').Total_EMI_per_month.std() \
    [train_df.groupby('Customer_ID').Total_EMI_per_month.std() > 300].index

# indic_3k_to_10k=df_Interest_emi_check.index[pd.DataFrame({'3000':df_Interest_emi_check.Total_EMI_per_month>3000 ,\
#                                                           '1000':df_Interest_emi_check.Total_EMI_per_month<10000}).all(axis=1)]
# cust2=df_Interest_emi_check.loc[indic_3k_to_10k,'Customer_ID'].unique()
# df_cust_3k_to10k=df_Interest_emi_check.where(df_Interest_emi_check.Customer_ID.isin(cust2)).dropna()

# emi_val_count=df_Interest_emi_check.value_counts()
# df_Interest_emi_check.describe()


df_Interest_emi_check = train_df.loc[:, ['Customer_ID', 'Total_EMI_per_month']][
    train_df.Customer_ID.isin(customer_emi_outlier)]
emi_invalid_values = df_Interest_emi_check[df_Interest_emi_check.Total_EMI_per_month > 1100].index

'''check annual income'''

customer_annual_out = train_df.groupby('Customer_ID').Annual_Income.std() \
    [train_df.groupby('Customer_ID').Annual_Income.std() > 10000].index

annual_invalid_values_ind = train_df.Annual_Income.index[
    train_df.Annual_Income.where(train_df.Customer_ID.isin(customer_annual_out)) > 200000]

'''check num credit inquiries'''
custid_numcredit_outlier=(train_df.groupby('Customer_ID').Num_Credit_Inquiries.std())\
    [train_df.groupby('Customer_ID').Num_Credit_Inquiries.std()>10].index

df_numcredit_check_outliers=train_df.iloc[:,:16].where(train_df.Customer_ID.isin(custid_numcredit_outlier)).dropna()


df_numcredit_out_percustomer=train_df.Num_Credit_Inquiries.index\
    [train_df.Num_Credit_Inquiries.where(train_df.Customer_ID.isin(custid_numcredit_outlier))>30]
'''check num of delyed payment'''
# num_delayed_q1=train_df.groupby('Customer_ID').Num_of_Delayed_Payment.quantile(q=0.25)
# num_delayed_q3=train_df.groupby('Customer_ID').Num_of_Delayed_Payment.quantile(q=0.75)
# num_delayed_iqr=num_delayed_q3-num_delayed_q1
# upper_fence=num_delayed_q3+1.5*num_delayed_iqr
# lower_fence=num_delayed_q1-1.5*num_delayed_iqr

custid_delayed_pay=train_df.groupby('Customer_ID').Num_of_Delayed_Payment.std()[train_df.groupby('Customer_ID').Num_of_Delayed_Payment.std() > 10].index

df_delay_pay_check=train_df[['Customer_ID','Num_of_Delayed_Payment']].where(train_df.Customer_ID.isin(custid_delayed_pay)).dropna()
sns.boxplot(train_df.Num_of_Delayed_Payment)
plt.show()
indicators_to_delayedpay=df_delay_pay_check.Num_of_Delayed_Payment[df_delay_pay_check.Num_of_Delayed_Payment>99].index


indicators_to_val = [emi_invalid_values, intr_rate_invalid_values, \
                                  annual_invalid_values_ind,df_numcredit_out_percustomer,indicators_to_delayedpay]
columns_to_updateval=['Total_EMI_per_month', 'Interest_Rate', 'Annual_Income','Num_Credit_Inquiries','Num_of_Delayed_Payment']
customer_ids=[customer_emi_outlier, customerid_inter_rate, customer_annual_out,custid_numcredit_outlier,custid_delayed_pay]

# start_index = None
# end_index = None
# for column, custids, indicator_list in tqdm(zip(columns_to_updateval,customer_ids,indicators_to_val)):
#     df = train_df.loc[:, ['Customer_ID', column]][train_df.Customer_ID.isin(custids)]
#     for indicator in indicator_list:
#         _ = df.loc[indicator, 'Customer_ID']
#         # if condition for reduce o() meaning reduce operations==>reduce time complexity
#         #         if int(indicator) - 8 > 0:
#         #             start_index = int(indicator) - 8
#         #         else:
#         #             start_index = 0
#         #         if int(indicator) + 8 < train_df.shape[0] - 1:
#         #             end_index = int(indicator) + 8
#         #         else:
#         #             end_index = train_df.shape[0] - 1
#
#         train_df.loc[indicator, column] = df.drop(indicator_list, axis=0) \
#                                               .loc[:, column][df['Customer_ID'] == _].mean()

# train_df.Num_of_Delayed_Payment.where(~(train_df.Num_of_Delayed_Payment<0),lambda x: np.abs(x),inplace=True)
sns.boxplot(train_df.Num_of_Delayed_Payment)
plt.show()
'''we can see that we have alot of outliers in annual income'''
# sns.boxplot(train_df.loc[:,'Annual_Income'])


# sns.histplot(train_df.loc[:,'Annual_Income'],log_scale=True)

# check outliers values for each customer annual income
# todo: check annual income
# customer_monthly_salary = train_df.groupby('Customer_ID').Monthly_Inhand_Salary.std()[
#     train_df.groupby('Customer_ID').Monthly_Inhand_Salary.std() > 100].index
# customer_Interest_Rate = train_df.groupby('Customer_ID').Interest_Rate.std()[
#     train_df.groupby('Customer_ID').Interest_Rate.std() > 100].index

# customer_emi_outlier = train_df.groupby('Customer_ID').Total_EMI_per_month.std()[
#     train_df.groupby('Customer_ID').Total_EMI_per_month.std() > 100].index

df_annual_income_check = train_df.loc[:, ['Customer_ID', 'Annual_Income']][
    train_df.Customer_ID.isin(customer_annual_out)]
df_annual_incomeandemi_check = \
    train_df.loc[:, ['Customer_ID', 'Annual_Income', 'Total_EMI_per_month', 'Monthly_Inhand_Salary' \
                        , 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Amount_invested_monthly', 'Interest_Rate',
                     'Monthly_Balance']] \
        [train_df.Customer_ID.isin(customer_annual_out)]
#
# df_salary = train_df.loc[:, ['Customer_ID', 'Monthly_Inhand_Salary']][
#     train_df.Customer_ID.isin(customer_monthly_salary)]
# df_salary_andmore_check = \
# train_df.loc[:, ['Customer_ID', 'Annual_Income', 'Total_EMI_per_month', 'Monthly_Inhand_Salary', 'Num_of_Loan' \
#                     , 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Amount_invested_monthly', 'Interest_Rate',
#                  'Monthly_Balance']] \
#     [train_df.Customer_ID.isin(customer_monthly_salary)]

# df_Interest_Rate = train_df.loc[:, ['Customer_ID', 'Interest_Rate']][train_df.Customer_ID.isin(customer_Interest_Rate)]
# df_Interest_Rate_andmore_check = \
# train_df.loc[:, ['Customer_ID', 'Annual_Income', 'Total_EMI_per_month', 'Monthly_Inhand_Salary', 'Num_of_Loan' \
#                     , 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Amount_invested_monthly', 'Interest_Rate',
#                  'Monthly_Balance']] \
#     [train_df.Customer_ID.isin(customer_Interest_Rate)]
#
#
#
#
#
# annual_income_scan = df_annual_incomeandemi_check.groupby('Customer_ID').Annual_Income.describe()
#
# pd.DataFrame(train_df.groupby('Customer_ID')['Interest_Rate'])
# train_df.groupby('Customer_ID').Interest_Rate
# train_df.groupby('Customer_ID').Total_EMI_per_month
# train_df.groupby('Customer_ID').Total_EMI_per_month.

columns_to_boxplot=train_df.columns[1:18]
columns_to_remove_outleirs=train_df.columns[2:18]
q1=None
q3=None
iqr=None
upper=None
lower=None


# for column in tqdm(columns_to_remove_outleirs):
#     q1=train_df.loc[:,column].quantile(0.25)
#     q3=train_df.loc[:, column].quantile(0.75)
#     iqr=q3-q1
#     upper=q3+2*iqr
#     lower=q1-2*iqr
#     train_df=train_df[train_df.loc[:,column]<upper]
#     train_df=train_df[train_df.loc[:,column]>lower]

train_df.to_csv('train_df.csv', index=False)


#
# for i in columns_to_boxplot:
#     sns.boxplot(train_df.loc[:,i])
#     plt.show()
#
# train_df2=train_df
# for column in tqdm(columns_to_remove_outleirs):
#     q1=train_df2.loc[:,column].quantile(0.25)
#     q3=train_df2.loc[:, column].quantile(0.75)
#     iqr=q3-q1
#     upper=q3+1.5*iqr
#     lower=q1-1.5*iqr
#     train_df2=train_df2[train_df2.loc[:,column]<upper]
#     train_df2=train_df2[train_df2.loc[:,column]>lower]
# print(train_df2.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline


# convert_dict={'Poor':0,'Standard':1,'Good':2}
# for (label,num) in convert_dict.items():
#     train_df2.loc[train_df2.index[train_df2.loc[:,'Credit_Score']==label],'Credit_Score']=num

# target=pd.get_dummies(train_df2.Credit_Score)
# x_train,x_test, y_train, y_test = train_test_split(train_df2.iloc[:,1:-1],target,test_size=0.25,stratify=target, random_state=42)
# modelknn=KNeighborsClassifier(n_neighbors=9,metric='manhattan',weights='distance')
# modeldecisiontree=DecisionTreeClassifier()
# modelrandom=RandomForestClassifier(n_estimators=64*3)
# modelgboost=GradientBoostingClassifier(n_estimators=64*3)
# models_selection=[modelknn,modelgboost,modelrandom,modeldecisiontree]
# pipeline=None
# score=None
# for model in models_selection:
#     pipeline=make_pipeline(StandardScaler(),model)
#     pipeline.fit(x_train, y_train)
#     score=cross_val_score(pipeline,x_train,y_train)
#     print(score.mean())
