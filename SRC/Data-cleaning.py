import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('ggplot')
from statistics import mean
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statsmodels.api as sm
import scipy as sp
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest

#import excel and create dataframes for officers and enlisted
ad_marital_status = pd.read_excel ('../Data/AD-by-marital-status.xls')
ad_marital_index = ad_marital_status.set_index('Pay Grade')

enlisted_df = ad_marital_status[:9]
enlisted_totals = enlisted_df.drop(columns=['Single Male', 'Single Female', 'Joint Service Marriage Male', 'Joint Service Marriage Female', \
    'Single Parent Male','Single Parent Female', 'Civilian Married Male','Civilian Married Female', 'Grand Total','Total Male', 'Total Female'])
officer_df = ad_marital_status[10:20]
officer_totals = officer_df.drop(columns=['Single Male', 'Single Female', 'Joint Service Marriage Male', 'Joint Service Marriage Female',\
    'Single Parent Male','Single Parent Female', 'Civilian Married Male','Civilian Married Female', 'Grand Total','Total Male', 'Total Female'])


def marital_status_plot_totals(df, enlisted=False):
    '''EDA plot of service members by rank and their marital status,
    takes in a data frame and whether the data frame contains enlisted or married soldiers'''
    x = df['Pay Grade']
    single_total= df['Single Total']
    single_parent_total = df['Single Parent Total']
    joint_service_marriage_total= df['Joint Service Marriage Total']
    civilian_married_total = df['Civilian Married Total']

    fig, ax = plt.subplots(figsize=(16,8)) 
    ax.plot(x, single_total, label = 'Single Total') 
    ax.plot(x, single_parent_total,label = 'Single Parent Total')
    ax.plot(x, joint_service_marriage_total,label = 'Joint Service Marriage Total')
    ax.plot(x,civilian_married_total ,label = 'Civilian Marriage Total')
    plt.legend(loc="upper right")
    plt.xlabel('Enlisted Rank')
    plt.ylabel('Number of Service Members')
    plt.rcParams.update({'font.size': 18})
    plt.grid()
    return plt.show()



def marital_status_bar_totals(df, enlisted=True):

    single_total= df['Single Total'].values
    single_parent_total= df['Single Parent Total'].values
    joint_service_marriage_total = df['Joint Service Marriage Total'].values
    civilian_married_total = df['Civilian Married Total'].values

    if enlisted==True:
        labels= ['E-1','E-2','E-3','E-4','E-5','E-6','E-7','E-8', 'E-9']
    elif enlisted==False:
        labels= ['O-1','0-2','O-3','O-4','O-5','O-6','O-7','O-8', 'O-9', 'O-10']

    x=np.arange(len(labels))
    width = 0.23

    fig, ax = plt.subplots(figsize=(20,8)) 
    rects1 = ax.bar(x, single_total ,width, label='Single Total',color=['red'])
    rects2 = ax.bar(x+width, single_parent_total ,width, label='Single Parent Total',color=['blue'])
    rects3 = ax.bar(x+2*width, joint_service_marriage_total ,width, label='Joint Service Married Total',color=['green'])
    rects4 = ax.bar(x+3*width, civilian_married_total ,width, label='Civilian Married Total',color=['black'])

   
    plt.xticks(x+width, labels)
    plt.legend(loc="upper right")
    plt.title('Service Members by Marital Status')
    plt.xlabel('Pay Grade')
    plt.ylabel('Number of Service Members')
    plt.rcParams.update({'font.size': 18})
    plt.grid()
    return plt.show()

def marital_status_bar_mean(df ):

    single_total = df['Single Total'].values
    single_parent_total = df['Single Parent Total'].values
    joint_service_marriage_total = df['Joint Service Marriage Total'].values
    civilian_married_total = df['Civilian Married Total'].values

    single_total_mean =np.mean(single_total)
    single_parent_total_mean =np.mean(single_parent_total)
    joint_service_marriage_total_mean = np.mean(joint_service_marriage_total)
    civilian_married_total_mean = np.mean(civilian_married_total)
    y = single_parent_total_mean, single_parent_total_mean, joint_service_marriage_total_mean, civilian_married_total_mean
    labels = ['Single', 'Single Parent', 'Joint Service Marriage', 'Civilian Marriage' ]

    x=np.arange(len(labels))
    width = 0.23

    fig, ax = plt.subplots(figsize=(20,8)) 
    plt.bar(x, y,color=['red', 'blue', 'green', 'black'] )
    

   
    plt.xticks(x, labels)
  
    plt.title('Service Members by Marital Status')
    plt.xlabel('Marital Status')
    plt.ylabel('Average Number of Service Members')
    plt.rcParams.update({'font.size': 18})
    return plt.show()

def ztest():
    ad_marital_index = ad_marital_status.set_index('Pay Grade')

    js_married_e = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Joint Service Marriage Total'].values)
    civ_married_e = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Civilian Married Total'].values)

    total_e = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Grand Total'].values)
    enlisted_married = int(js_married_e) + int(civ_married_e)
    married_prop_e = enlisted_married/total_e

    ad_marital_index = ad_marital_status.set_index('Pay Grade')
    js_married_o = int(ad_marital_index.loc[['TOTAL OFFICER'],'Joint Service Marriage Total'].values)
    civ_married_o = int(ad_marital_index.loc[['TOTAL OFFICER'],'Civilian Married Total'].values)

    total_o = int(ad_marital_index.loc[['TOTAL OFFICER'],'Grand Total'].values)
    officer_married = int(js_married_o) + int(civ_married_o)
    married_prop_o = officer_married/total_o

    significance = 0.025

    sample_prop_e, sample_size_e = (married_prop_e, total_e)
    sample_prop_o, sample_size_o = (married_prop_o, total_o)

    successes = np.array([sample_prop_e, sample_prop_o])
    samples = np.array([sample_size_e, sample_size_o])


    stat, p_value = proportions_ztest(count=successes, nobs=samples,  alternative='two-sided')

    print('z_stat: %0.3f, p_value: %0.3f' % (stat, p_value))

    if p_value > significance:
        print("Fail to reject the null hypothesis - we have nothing else to say")
    else:
        print("Reject the null hypothesis - suggest the alternative hypothesis is true")
        


#if __name__ == '__main__':
    
    #marital_status_plot_totals(enlisted_totals, enlisted=True)

    #marital_status_bar_totals(officer_df, False )
    
    #marital_status_bar_mean(officer_totals)
    