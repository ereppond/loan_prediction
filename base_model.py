import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

num_train = train.copy()
num_test = test.copy()

def fillna(df1, df2, columns, values):
    for i, col in enumerate(columns):
        df1[col].fillna(values[i], inplace=True)
        df2[col].fillna(values[i], inplace=True)
    return df1, df2

num_train, num_test = fillna(num_train, 
                             num_test, 
                             ['Gender', 'Married', 'Education',
                              'Self_Employed', 'Property_Area',
                              'Dependents', 'Credit_History', 
                              'LoanAmount', 'Loan_Amount_Term'],
                             ['None', 'None', 'None', 'None', 'None', 
                              -1, -1, -1, -1])

def label_encoder(df1, df2, columns):
    for col in columns:
        cur = LabelEncoder()
        df1[col] = cur.fit_transform(df1[col])
        df2[col] = cur.transform(df2[col])
    return df1, df2

num_train, num_test = label_encoder(num_train, num_test, ['Gender', 'Married', 'Education', 
                                                          'Self_Employed', 'Property_Area'])

def change_dep(df1, df2):
    df1['Dependents'] = df1['Dependents'].apply(lambda x: 3 if str(x) == '3+' else int(x))
    df2['Dependents'] = df2['Dependents'].apply(lambda x: 3 if str(x) == '3+' else int(x))
    return df1, df2

num_train, num_test = change_dep(num_train, num_test)

ids = list(num_train['Loan_ID'].values)
ids.extend(num_test['Loan_ID'].values)
ID = LabelEncoder()
ID.fit_transform(ids)
num_train['Loan_ID'] = ID.transform(num_train['Loan_ID'])
num_test['Loan_ID'] = ID.transform(num_test['Loan_ID'])

X_train, X_test, y_train, y_test = train_test_split(num_train.drop('Loan_Status', axis=1), 
    num_train['Loan_Status'], random_state=42, test_size=.25)


rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf.score(X_test, y_test)


if __name__ == '__main__':
    return None



