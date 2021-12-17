import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

ad_marital_status = pd.read_excel ('../Data/AD-by-marital-status.xls')
enlisted_df = ad_marital_status[:9]
enlisted_totals = enlisted_df.drop(columns=['Single Male', 'Single Female', 'Joint Service Marriage Male', 'Joint Service Marriage Female', 'Single Parent Male','Single Parent Female', 'Civilian Married Male','Civilian Married Female', 'Grand Total','Total Male', 'Total Female'])
officer_df = ad_marital_status[10:20]
officer_totals = officer_df.drop(columns=['Single Male', 'Single Female', 'Joint Service Marriage Male', 'Joint Service Marriage Female','Single Parent Male','Single Parent Female', 'Civilian Married Male','Civilian Married Female', 'Grand Total','Total Male', 'Total Female'])


def marital_status_plot_totals(df, enlisted=False):

    single_total= df['Single Total'].values
    single_parent_total= df['Single Parent Total'].values
    joint_service_marriage_total = df['Joint Service Marriage Total'].values
    civilian_married_total = df['Civilian Married Total'].values

    if enlisted==True:
        labels= ['E-1','E-2','E-3','E-4','E-5','E-6','E-7','E-8', 'E-9']
    elif enlisted==False:
        labels= ['O-1','0-2','O-3','O-4','O-5','O-6','O-7','O-8', 'O-9', 'O-10']    

    x = officer_totals['Pay Grade']
    fig, ax = plt.subplots(figsize=(16,8)) 
    ax.plot(x, officer_totals['Single Total'], label = 'Single Total') 
    ax.plot(x, officer_totals['Single Parent Total'],label = 'Single Parent Total')
    ax.plot(x, officer_totals['Joint Service Marriage Total'],label = 'Joint Service Marriage Total')
    ax.plot(x, officer_totals['Civilian Married Total'],label = 'Civilian Marriage Total')
    plt.legend(loc="upper right")
    plt.title('Service Members by Marital Status')
    plt.xlabel('Pay Grade')
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
    rects1 = ax.bar(x, single_total ,width, label='Single Total')
    rects2 = ax.bar(x+width, single_parent_total ,width, label='Single Parent Total')
    rects3 = ax.bar(x+2*width, joint_service_marriage_total ,width, label='Joint Service Married Total')
    rects4 = ax.bar(x+3*width, civilian_married_total ,width, label='Civilian Married Total')

   
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
    rects1 = ax.bar(x, y )
    

   
    plt.xticks(x, labels)
  
    plt.title('Service Members by Marital Status')
    plt.xlabel('Marital Status')
    plt.ylabel('Number of Service Members')
    plt.rcParams.update({'font.size': 18})
   
    return plt.show()

    
if __name__ == '__main__':
    
    #marital_status_bar_totals(enlisted_totals)

    marital_status_bar_mean(officer_df )