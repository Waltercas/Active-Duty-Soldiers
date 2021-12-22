#Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('ggplot')
from statistics import mean
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statsmodels.api as sm
import scipy as sp
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest


#Read excel and create dataframes for officers and enlisted
ad_marital_status = pd.read_excel ('../Data/AD-by-marital-status.xls')
ad_marital_index = ad_marital_status.set_index('Pay Grade')

enlisted_df = ad_marital_status[:9]
enlisted_totals = enlisted_df.drop(columns=['Single Male', 'Single Female', 'Joint Service Marriage Male', 'Joint Service Marriage Female', \
    'Single Parent Male','Single Parent Female', 'Civilian Married Male','Civilian Married Female', 'Grand Total','Total Male', 'Total Female'])
officer_df = ad_marital_status[10:20]
officer_totals = officer_df.drop(columns=['Single Male', 'Single Female', 'Joint Service Marriage Male', 'Joint Service Marriage Female',\
    'Single Parent Male','Single Parent Female', 'Civilian Married Male','Civilian Married Female', 'Grand Total','Total Male', 'Total Female'])


def plot_totals_rank(df,save_bool, plot_bool, figname= 'Marrital Status Multiple Plot By Rank'):
    '''EDA plot of service members by rank and their marital status,
    takes in a data frame and whether the data frame contains enlisted or married soldiers
    takes in bools to determine if you would like to view, save or both'''
   
    #Create dataframes
    x = df['Pay Grade']
    single_total= df['Single Total']
    single_parent_total = df['Single Parent Total']
    joint_service_marriage_total= df['Joint Service Marriage Total']
    civilian_married_total = df['Civilian Married Total']

    #Create plot
    fig, ax = plt.subplots(figsize=(16,6)) 
    plt.rcParams.update({'font.size': 18})
    
    ax.plot(x, single_total, label = 'Single Total') 
    ax.plot(x, single_parent_total,label = 'Single Parent Total')
    ax.plot(x, joint_service_marriage_total,label = 'Joint Service Marriage Total')
    ax.plot(x,civilian_married_total ,label = 'Civilian Marriage Total')
    plt.legend(loc="upper right")
    plt.title('Service Members By Marital Status')
    plt.xlabel('Rank')
    plt.ylabel('Number of Service Members')
    
    if save_bool == 1:
        plt.savefig(figname)
    if plot_bool == 1:
        plt.show()
        plt.grid()



def bar_totals_rank(df, save_bool, plot_bool, enlisted=True, figname= 'Marrital Status Multiple Bar Plot By Rank'):
    """Multiple bar plots that display a plot of the marital status totals by rank
    takes in bools to determine if you would like to view, save or both"""

    single_total= df['Single Total'].values
    single_parent_total= df['Single Parent Total'].values
    joint_service_marriage_total = df['Joint Service Marriage Total'].values
    civilian_married_total = df['Civilian Married Total'].values

    if enlisted==True:
        labels= ['E-1','E-2','E-3','E-4','E-5','E-6','E-7','E-8', 'E-9']
        rank = "Officer"
    elif enlisted==False:
        labels= ['O-1','0-2','O-3','O-4','O-5','O-6','O-7','O-8', 'O-9', 'O-10']
        rank = "Enlisted"

    x=np.arange(len(labels))
    width = 0.23

    fig, ax = plt.subplots(figsize=(16,6)) 
    plt.rcParams.update({'font.size': 18})

    rects1 = ax.bar(x, single_total ,width, label='Single Total',color=['red'])
    rects2 = ax.bar(x+width, single_parent_total ,width, label='Single Parent Total',color=['blue'])
    rects3 = ax.bar(x+2*width, joint_service_marriage_total ,width, label='Joint Service Married Total',color=['green'])
    rects4 = ax.bar(x+3*width, civilian_married_total ,width, label='Civilian Married Total',color=['black'])


    plt.rcParams.update({'font.size': 18})
    plt.xticks(x+width, labels)
    plt.legend(loc="upper right")
    plt.title('{} Service Members by Marital Status'.format(rank))
    plt.xlabel('Pay Grade')
    plt.ylabel('Number of Service Members')
    
    if save_bool == 1:
        plt.savefig(figname)
    if plot_bool == 1:
        plt.show()
        plt.grid()
        
def bar_means(df,save_bool,plot_bool, figname = 'Marital Status Mean By Rank'):
    """Function that takes a data frame and creates a bar plot of the means of soldiers marital status
    one for officers and enlisted
    takes in bools to determine if you would like to view, save or both"""

    single_total = df['Single Total'].values
    single_parent_total = df['Single Parent Total'].values
    joint_service_marriage_total = df['Joint Service Marriage Total'].values
    civilian_married_total = df['Civilian Married Total'].values

    single_total_mean =np.mean(single_total)
    single_parent_total_mean =np.mean(single_parent_total)
    joint_service_marriage_total_mean = np.mean(joint_service_marriage_total)
    civilian_married_total_mean = np.mean(civilian_married_total)
    y = single_total_mean, single_parent_total_mean, joint_service_marriage_total_mean, civilian_married_total_mean
    labels = ['Single', 'Single Parent', 'Joint Service Marriage', 'Civilian Marriage' ]

    x=np.arange(len(labels))
    width = 0.23

    fig, ax = plt.subplots(figsize=(16,6)) 
    plt.rcParams.update({'font.size': 18})
    plt.bar(x, y,color=['red', 'blue', 'green', 'black'] )
    
    plt.xticks(x, labels)
    if len(df.index) > 9:
        rank = "Officers"
    else:
        rank = "Enlisted"
        
    plt.title('{} by Marital Status'.format(rank))
    plt.xlabel('Marital Status')
    plt.ylabel('Average Number of Service Members')
    
    if save_bool == 1:
        plt.savefig(figname)
    if plot_bool == 1:
        plt.show()
        plt.grid() 

def ad_bar_mean(save_bool, plot_bool, ):
    '''Function that creates bar plot of mean of officer or enlisted married personel by rank
    takes in a dataframe takes in bools to determine if you would like to view, save or both'''

    single_total = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Single Total'].values) + int(ad_marital_index.loc[['TOTAL OFFICER'],'Single Total'].values)
    single_parent_total = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Single Parent Total'].values) + int(ad_marital_index.loc[['TOTAL OFFICER'],'Single Parent Total'].values)
    joint_service_marriage_total = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Joint Service Marriage Total'].values) + int(ad_marital_index.loc[['TOTAL OFFICER'],'Joint Service Marriage Total'].values)
    civilian_married_total = int(ad_marital_index.loc[['TOTAL OFFICER'],'Civilian Married Total'].values) + int(ad_marital_index.loc[['TOTAL ENLISTED'],'Civilian Married Total'].values)


    single_total_mean =np.mean(single_total)
    single_parent_total_mean =np.mean(single_parent_total)
    joint_service_marriage_total_mean = np.mean(joint_service_marriage_total)
    civilian_married_total_mean = np.mean(civilian_married_total)
    y = single_total_mean, single_parent_total_mean, joint_service_marriage_total_mean, civilian_married_total_mean
    labels = ['Single', 'Single Parent', 'Joint Service Marriage', 'Civilian Marriage' ]

    x=np.arange(len(labels))
    width = 0.23

    fig, ax = plt.subplots(figsize=(16,6))
    plt.rcParams.update({'font.size': 18})
    plt.bar(x, y,color=['red', 'blue', 'green', 'black'] )
    
    plt.xticks(x, labels)
  
    plt.title('Service Members by Marital Status')
    plt.xlabel('Marital Status')
    plt.ylabel('Average Number of Service Members')
    
    if save_bool == 1:
        plt.savefig('AD Marrital Status Mean')
    if plot_bool == 1:
        plt.show()
        plt.grid()
    
def ad_pie_chart(save_bool, plot_bool):
    single_total = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Single Total'].values) + int(ad_marital_index.loc[['TOTAL OFFICER'],'Single Total'].values)
    single_parent_total = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Single Parent Total'].values) + int(ad_marital_index.loc[['TOTAL OFFICER'],'Single Parent Total'].values)
    joint_service_marriage_total = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Joint Service Marriage Total'].values) + int(ad_marital_index.loc[['TOTAL OFFICER'],'Joint Service Marriage Total'].values)
    civilian_married_total = int(ad_marital_index.loc[['TOTAL OFFICER'],'Civilian Married Total'].values) + int(ad_marital_index.loc[['TOTAL ENLISTED'],'Civilian Married Total'].values)

    y= np.array([single_total, single_parent_total, joint_service_marriage_total, civilian_married_total])
    mylabels = ["Single", "Single Parent", "Joint Service Married", "Civilian Married"]

    plt.figure(figsize = (16,6))
    plt.rcParams.update({'font.size': 18})
    plt.pie(y, labels = mylabels, autopct = "%0.1f%%" ,colors=['red', 'blue', 'green', 'orange'])
    plt.grid()
    if save_bool == 1:
        plt.savefig('AD Pie Chart')
    if plot_bool == 1:
        plt.show()
        plt.grid()

def ztest_variables():

    global married_prop_e, married_prop_o, married_prop, unmarried_prop_e, unmarried_prop_o, unmarried_prop, total_o, total_sm, total_e, total_unmarried, total_married
    ad_marital_index = ad_marital_status.set_index('Pay Grade')

    single_e = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Single Total'].values)
    single_parent_e = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Single Parent Total'].values)
    js_married_e = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Joint Service Marriage Total'].values)
    civ_married_e = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Civilian Married Total'].values)
    total_e = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Grand Total'].values)
    enlisted_married = int(js_married_e) + int(civ_married_e)
    enlisted_unmarried = single_parent_e + single_e


    single_o = int(ad_marital_index.loc[['TOTAL OFFICER'],'Single Total'].values)
    single_parent_o = int(ad_marital_index.loc[['TOTAL OFFICER'],'Single Parent Total'].values)
    js_married_o = int(ad_marital_index.loc[['TOTAL OFFICER'],'Joint Service Marriage Total'].values)
    civ_married_o = int(ad_marital_index.loc[['TOTAL OFFICER'],'Civilian Married Total'].values)
    total_o = int(ad_marital_index.loc[['TOTAL OFFICER'],'Grand Total'].values)
    officer_married = int(js_married_o) + int(civ_married_o)
    officer_unmarried = single_parent_o + single_o


    total_e = int(ad_marital_index.loc[['TOTAL ENLISTED'],'Grand Total'].values)
    total_o = int(ad_marital_index.loc[['TOTAL OFFICER'],'Grand Total'].values)
    total_sm =total_e + total_o
    total_married = enlisted_married + officer_married
    total_unmarried = enlisted_unmarried + officer_unmarried
    unmarried_prop_e = enlisted_unmarried/total_e
    unmarried_prop_o = officer_unmarried/total_o
    unmarried_prop = total_unmarried / total_sm
    married_prop_o = officer_married/total_o
    married_prop_e = enlisted_married/total_e
    married_prop = total_married / total_sm


def test(proportion_A, proportion_B, population_A, population_B):
    ztest_variables()
    significance = 0.05

    sample_prop_A, sample_size_A = (proportion_A, population_A)
    sample_prop_B, sample_size_B = (proportion_B, population_B)

    successes = np.array([sample_prop_A, sample_prop_B])
    samples = np.array([sample_size_A, sample_size_B])


    stat, p_value = proportions_ztest(count=successes, nobs=samples,  alternative='two-sided')

    print('z_stat: %0.3f, p_value: %0.3f' % (stat, p_value))

    if p_value > significance:
        statement = print("Fail to reject the null hypothesis - we have nothing else to say")
    else:
        statement = print("Reject the null hypothesis - suggest the alternative hypothesis is true")
        
    return statement       


if __name__ == '__main__':

    """Fig 1-2 plots enlisted and officers marital status by rank for EDA"""
    #plot_totals_rank(enlisted_df,1,1, figname= 'Marrital Status Multiple Plot Enlisted')
    #plot_totals_rank(officer_df,1,1,figname= 'Marrital Status Multiple Plot Officers')

    """Fig 3-4 multiple bar plots of total number of service members by marital status by rank  """
    #bar_totals_rank(enlisted_df,1,1, enlisted=True, figname= 'Marrital Status Multiple Bar Plot By Enlisted')
    #bar_totals_rank(officer_df,1,1, enlisted=False,figname= 'Marrital Status Multiple Bar Plot By Officers')

    """Fig 5-6 bar plots of total number of service members by marital status split by officer and enlisted """
    #bar_means(enlisted_df,1,1,figname = 'Marital Status Mean Enlisted')
    #bar_means(officer_df,1,1,figname = 'Marital Status Mean By Officer')

    """Fig 7 bar plot all active duty """
    #ad_bar_mean(1,1)

    """Creates variables for the z test"""
    #ztest_variables()

    """Z-tests (must run with variables"""


    #test(married_prop_e, married_prop_o, total_e, total_o)
    #test(unmarried_prop_e, unmarried_prop_o, total_e, total_o)
    #test(married_prop, unmarried_prop, total_sm, total_sm)

    #ad_pie_chart(1,1)


