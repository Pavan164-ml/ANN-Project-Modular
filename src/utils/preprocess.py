import pandas as pd
import warnings
import logging
from sklearn.preprocessing import MinMaxScaler  
warnings.filterwarnings("ignore")

def preprocessor(df):
    df.drop('customerID',axis='columns',inplace=True)

    # Where ever there is ' ' in TotalCharges it is ignored and rest is stored in df1
    df = df[df.TotalCharges!=' ']
    df.TotalCharges = pd.to_numeric(df.TotalCharges)
    df.replace('No internet service','No',inplace=True)
    df.replace('No phone service','No',inplace=True)

    ## These are all the columns with yes and no
    yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
    
    ## Applying the for loop for 1's and 0's in place of yes and no
    for col in yes_no_columns:
        df[col].replace({'Yes': 1,'No': 0},inplace=True)

    ## Similarly for male and female
    df['gender'].replace({'Female':1,'Male': 0},inplace=True)

    ## For all the columns which has more than 2 classes we apply get_dummies
    data1 = pd.get_dummies(data=df , columns=['InternetService','Contract','PaymentMethod'])

    ## Normalizing data using MinMax scaler
    columns_to_be_scaled = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = MinMaxScaler()
    data1[columns_to_be_scaled] = scaler.fit_transform(data1[columns_to_be_scaled])


    pd.set_option('display.max_columns', None)
    print(data1.head(2))
    return data1
