import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# path = r'mini_project/train.csv'
path = r'train.csv'
df = pd.read_csv(path, low_memory=False)

# df.info()
df.groupby([df.Customer_ID, df.Age])


def frequency(data, column_name):
    return data[column_name].groupby([data.Customer_ID]).describe()


print('1')

list_columns_clean_ = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Amount_invested_monthly'
                       , 'Outstanding_Debt', 'Monthly_Balance']
for column in list_columns_clean_:
    df[column] = df[column].str.replace('_', '')

df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].str.replace('_', '0')
df['Num_Bank_Accounts'] = df['Num_Bank_Accounts'].replace(-1, 0)
df['Num_of_Loan'] = df['Num_of_Loan'].replace(-100, 0)

describe = df.describe()
# # this columns not need clean
# # 0 clean columns ID
# # 1 clean columns Customer_ID
# # 2 clean columns Month
# # 3 clean columns Name
#
#
# # 4 clean columns Age
# age_groupby = frequency(df, 'Age')
# for i in tqdm(range(len(df))):
#     if len(df['Age'][i]) != 2:
#         df['Age'][i] = age_groupby['top'].loc[df['Customer_ID'][i]]
#     else:
#         continue
#
# # 5 clean columns SSN
# SSN_groupby = frequency(df, 'SSN')
# for i in tqdm(range(len(df))):
#     if list(df['SSN'][i])[0].isnumeric() is False:
#         df['SSN'][i] = SSN_groupby['top'].loc[df['Customer_ID'][i]]
#     else:
#         continue
#
# # 6 clean columns Occupation
# Occupation_groupby = frequency(df, 'Occupation')
# for i in tqdm(range(len(df))):
#     if list(df['Occupation'][i])[0].isalpha() is False:
#         df['Occupation'][i] = Occupation_groupby['top'].loc[df['Customer_ID'][i]]
#     else:
#         continue
#
# # 7 clean columns Annual_Income
# # this column not need clean
#
# # 8 clean columns Monthly_Inhand_Salary
# # todo fill null : Monthly_Inhand_Salary_null =df[df['Monthly_Inhand_Salary'].isnull()]
#
#
# index_null_list = list(df.index[df['Monthly_Inhand_Salary'].isnull()])
# for i in tqdm(index_null_list):
#     df.loc[i, 'Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'][df['Customer_ID'] == df['Customer_ID'][i]].mode()[
#         0]
#
# # 9 clean columns Num_Bank_Accounts
# Num_Bank_Accounts_groupby = frequency(df, 'Num_Bank_Accounts')
# q4 = Num_Bank_Accounts_groupby['max'].loc[df['Customer_ID']]
# q3 = Num_Bank_Accounts_groupby['75%'].loc[df['Customer_ID']]
#
# for i in tqdm(range(len(df))):
#     if [q4[i] != q3[i]] == [True]:
#         df['Num_Bank_Accounts'][i] = Num_Bank_Accounts_groupby['min'].loc[df['Customer_ID'][i]]
#     else:
#         continue
#
# # 10 clean columns Num_Credit_Card
# Num_Credit_Card_groupby = frequency(df, 'Num_Credit_Card')
# q4 = Num_Credit_Card_groupby['max'].loc[df['Customer_ID']]
# q3 = Num_Credit_Card_groupby['75%'].loc[df['Customer_ID']]
#
# for i in tqdm(range(len(df))):
#     if [q4[i] != q3[i]] == [True]:
#         df['Num_Credit_Card'][i] = Num_Credit_Card_groupby['min'].loc[df['Customer_ID'][i]]
#     else:
#         continue
#
# # 11 clean columns Interest_Rate
# Interest_Rate_groupby = frequency(df, 'Interest_Rate')
# q4 = Interest_Rate_groupby['max'].loc[df['Customer_ID']]
# q3 = Interest_Rate_groupby['75%'].loc[df['Customer_ID']]
#
# for i in tqdm(range(len(df))):
#     if [q4[i] != q3[i]] == [True]:
#         df['Interest_Rate'][i] = Interest_Rate_groupby['min'].loc[df['Customer_ID'][i]]
#     else:
#         continue
#
# # 12 clean columns Num_of_Loan
# df['Num_of_Loan'] = df['Num_of_Loan'].astype('float32')
# Num_of_Loan_groupby = frequency(df, 'Num_of_Loan')
# q4 = Num_of_Loan_groupby['max'].loc[df['Customer_ID']]
# q3 = Num_of_Loan_groupby['75%'].loc[df['Customer_ID']]
#
# for i in tqdm(range(len(df))):
#     if [q4[i] != q3[i]] == [True]:
#         df['Num_of_Loan'][i] = Num_of_Loan_groupby['min'].loc[df['Customer_ID'][i]]
#     else:
#         continue
#
# # 13 clean columns Type_of_Loan
# # todo unique series and get_dummies all kind loans
#
# # 14 clean columns Delay_from_due_date
# # this column not need clean
#
# # 15 clean columns Num_of_Delayed_Payment
#
# df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].astype('float32')
# Num_of_Delayed_Payment_groupby = frequency(df, 'Num_of_Delayed_Payment')
# mean_d_p = df['Num_of_Delayed_Payment'].mean()
# q4 = Num_of_Delayed_Payment_groupby['max'].loc[df['Customer_ID']]
#
# for i in tqdm(range(len(df))):
#     if [q4[i] > mean_d_p] == [True]:
#         df['Num_of_Delayed_Payment'][i] = Num_of_Delayed_Payment_groupby['75%'].loc[df['Customer_ID'][i]]
#     else:
#         continue
#
# # 16 clean columns Changed_Credit_Limit
#
# # 17 clean columns Num_Credit_Inquiries
#
# index_null_list = list(df.index[df['Num_Credit_Inquiries'].isnull()])
# for i in tqdm(index_null_list):
#     df.loc[i, 'Num_Credit_Inquiries'] = df['Num_Credit_Inquiries'][df['Customer_ID'] == df['Customer_ID'][i]].mode()[0]
#
# df['Num_Credit_Inquiries'] = df['Num_Credit_Inquiries'].astype('float32')
# Num_Credit_Inquiries_groupby = frequency(df, 'Num_Credit_Inquiries')
# mean_n_c_i = df['Num_Credit_Inquiries'].mean()
# q4 = Num_Credit_Inquiries_groupby['max'].loc[df['Customer_ID']]
#
# for i in tqdm(range(len(df))):
#     if [q4[i] > mean_n_c_i] == [True]:
#         df['Num_Credit_Inquiries'][i] = Num_Credit_Inquiries_groupby['75%'].loc[df['Customer_ID'][i]]
#     else:
#         continue
# # 18 clean columns Credit_Mix
# df['Credit_Mix'] = df['Credit_Mix'].str.replace('_', '')
# df['Credit_Mix'] = df['Credit_Mix'].replace('', None)
# index_null_list = list(df.index[df['Credit_Mix'].isnull()])
# # index_null_list = list(df.index[df['Credit_Mix'] == '_'])
# # index_null_list_b = list(df.index[df['Credit_Mix'] == ''])
# # index_null_list += index_null_list_b
# for i in tqdm(index_null_list):
#     df.loc[i, 'Credit_Mix'] = df['Credit_Mix'][df['Customer_ID'] == df['Customer_ID'][i]].mode()[0]
#
#
# # 19 clean columns Outstanding_Debt
## 'not need to clean'
# # 20 clean columns Credit_Utilization_Ratio
#
# # 21 clean columns Credit_History_Age
#
#
# index_null_list = list(df.index[df['Credit_History_Age'].isnull()])
# for i in tqdm(range(len(df))):
#     if i not in index_null_list:
#         _ = (df['Credit_History_Age'][i].split())
#         df.loc[i, 'Credit_History_Age'] = int(_[0])*12 + int(_[3])
#     else:
#         continue
#
# for i in tqdm(index_null_list):
#     if i % 8 != 0:
#         df.loc[i, 'Credit_History_Age'] = df['Credit_History_Age'][i-1] + 1
#     else:
#         df.loc[i, 'Credit_History_Age'] = df['Credit_History_Age'][i+1] - 1
#
# # # # todo check mising values this fun is idea
# # index_null_list = list(df.index[df['Credit_History_Age'].isnull()])
# # for i in tqdm(index_null_list):
# #     if i % 8 == 0:
# #         df.loc[i, 'Credit_History_Age'] = df['Credit_History_Age'][i+4] - 4

# # 22 clean columns Payment_of_Min_Amount
#
# # 23 clean columns Total_EMI_per_month
#
# # 24 clean columns Amount_invested_monthly
#
# # 25 clean columns Payment_Behaviour
#
# # todo check mising values
# # 26 clean columns Monthly_Balance
#
# # 27 clean columns Credit_Score


df['Age'] = df['Age'].astype('float32')
df['Annual_Income'] = df['Annual_Income'].astype('float32')
df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].astype('float32')
df['Num_of_Loan'] = df['Num_of_Loan'].astype('float32')
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].astype('float32')
df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].astype('float32')
df['Amount_invested_monthly'] = df['Amount_invested_monthly'].astype('float32')
df['Num_of_Loan'] = df['Num_of_Loan'].astype('float32')
df['Delay_from_due_date'] = df['Delay_from_due_date'].astype('float32')
df['Outstanding_Debt'] = df['Outstanding_Debt'].astype('float32')
# df['Credit_History_Age'] = df['Credit_History_Age'].astype('float32')
df['Monthly_Balance'] = df['Monthly_Balance'].astype('float32')

describe = df.describe()
int_row = list(describe[0:0])
print('7')
df.columns
# df.info()


# sns.pairplot(a)
# plt.waitforbuttonpress()
# plt.show()

# for i,c in enumerate(df.columns):
#     print(f'#{i} clean columns {c} ')
# sns.histplot(data=df['Num_of_Delayed_Payment'])
# # sns.lineplot(df.Num_of_Delayed_Payment, )
# plt.waitforbuttonpress()
# plt.show()

## export as csv_file
# csv_df = df.to_csv( index=False)
# with open('csv_df.csv', 'w') as csv_file:
#     csv_df.to_csv(path_or_buf=csv_file)

