# Load libraries
import pandas
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from itertools import groupby
import os
from sklearn.covariance import EmpiricalCovariance, MinCovDet,EllipticEnvelope
# import modules
import pandas as pd
 
def prepare_variables(series):
    if (series['credit/debit']) == 0:
       
       
        series = series.set_value('totalnoofdebits',series.at['transaction_count'])  
        series = series.set_value('totaldebit',series.at['amount'])
        series = series.set_value('totalnoofcredits',0.0)  
        series = series.set_value('totalcredit',0.0)
        
    elif (series['credit/debit']) == 1 :
       
        series = series.set_value('totalnoofdebits',0.0)  
        series = series.set_value('totaldebit',0.0)
        series = series.set_value('totalnoofcredits',series.at['transaction_count'])  
        series = series.set_value('totalcredit',series.at['amount'])
        
    return series

 
# Load dataset
#url = "D:\\mycode\\python\\machine_learning\\iris.data"
url =  os.path.join('Dataset','transactiondata.dat')
#url =  os.path.join('Dataset','bank.data')
url1 =os.path.join('Dataset', 'customermaster.dat')
names = ['cust_id', 'credit/debit', 'amount', 'date', 'status']
names1 = ['cust_name', 'salary','cust_id']
dataset = pandas.read_csv(url, names=names)
dataset1 = pandas.read_csv(url1, names=names1)


#Get transaction data and master data from csv files
trasacationdata = pd.DataFrame(dataset, columns = ['cust_id', 'credit/debit', 'amount', 'date', 'status'])
masterdata = pd.DataFrame(dataset1, columns = ['cust_name', 'salary','cust_id'])


###############################################################
#To analyse on the bank transaction data, we will first prepare the data to represent following variables
#For each customer id,we will aggregate the following variables
#total transactio amount( credit/debit wise)
#total transaction count ( credit/debit wise)
# salary
#
#We will predict outliers based on the above variables. 
#
###############################################################
##aggregate transaction data , based on multi index - customer id, credit / debit
transactionsummary_results = trasacationdata.groupby(['cust_id', 'credit/debit']).sum().reset_index()
##get transaction 

agg_results = pd.DataFrame({'transaction_count':trasacationdata.groupby(['cust_id', 'credit/debit'],as_index=False).size()}).reset_index()
agg_results.to_csv('agg_results')
merge_finaldataset = pd.merge(transactionsummary_results,agg_results,on=['cust_id', 'credit/debit'])
#############################################################
# Testing set for the model would be chosen as the complete dataset
# Although the dataset is contaminated, we would use elliptical envelope on the complete transaction date
# by choosing appropriate contamination factor
############################################################

#create a dataset with variables mentioned about
merge_finaldataset_trim = merge_finaldataset[['cust_id', 'credit/debit','amount','transaction_count']]
merge_finaldataset_trim_salary = pd.merge(merge_finaldataset_trim,masterdata,how='outer',on=['cust_id'])
###########################################################
## Get the set grouped with no of transactions and transaction amount per customer
#zero_series = pd.Series(np.zeros(len(merge_finaldataset_trim_salary)))
#merge_finaldataset_trim_salary= merge_finaldataset_trim_salary.assign(totaldebit=zero_series,totalcredit=zero_series,totalnoofdebits=zero_series,totalnoofcredits=zero_series)
merge_finaldataset_trim_salary= merge_finaldataset_trim_salary.apply(prepare_variables, axis='columns',broadcast=False)
merge_finaldataset_trim_salary= merge_finaldataset_trim_salary.groupby('cust_id').sum().reset_index()
#merge_finaldataset_trim_salary['totaldebit','totalcredit','totalnoofdebits','totalnoofcredits'] =pd.Series(np.zeros(len(merge_finaldataset_trim_salary.index)))
merge_finaldataset_trim_salary.to_csv("mergettotaldebit")

merge_finaldataset_trim_salary_withoutcustid = merge_finaldataset_trim_salary[['totalnoofdebits','totaldebit' ,'totalnoofcredits','totalcredit','salary']] #customer id is not actually a variable
merge_finaldataset_trim_salary_withoutcustid.to_csv('mergefinal')
print(merge_finaldataset_trim_salary_withoutcustid.describe())
#we have the final data set with us, with dimension salary added to transaction data
#With salary dimension added, this data set is assumed to be Gaussian normal distribution 
# With this assumption, and since we have a contaminated data set , we will do a outliers analysis on the dataset
#We will fit the data points in a elliptic envelope, and try to predict outliers on a selected data set
print("Starting fit ... ")
outlier_analysis = EllipticEnvelope(contamination=0.5).fit(merge_finaldataset_trim_salary_withoutcustid)
print("Fit completed...")

print('Plotting Graph .....')
plt.figure(1)
plt.scatter(merge_finaldataset_trim_salary_withoutcustid['totaldebit'],merge_finaldataset_trim_salary_withoutcustid['totalcredit'],c='b')
plt.show()
# predict if the dataset is valid, by taking a testing set with unusual transactions
testing_set = pd.DataFrame( {'cust_id' : pd.Series([42], index=[0]),
'totalnoofdebits' : pd.Series([4], index=[0]),
                   'totaldebit': pd.Series([1000],index=[0]),
                   'totalnoofcredits':pd.Series([500],index=[0]),
                   'totalcredit':pd.Series([120000],index=[0]),
                   'salary':pd.Series([120000],index=[0])
                   
                   })
print(testing_set)
print("going to predict")
outlier_analysis.predict(testing_set[['totalnoofdebits','totaldebit','totalnoofcredits','totalcredit','salary']]) # again, customer id is not actually a variable
plt.close()               


