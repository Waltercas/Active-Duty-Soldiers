import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
 
ad_marital_status = pd.read_excel ('Data/AD-by-marital-status.xls')
enlisted_df = ad_marital_status[:9]
enlisted_df2 = enlisted_df.drop(columns=['Single Total', 'Joint Service Marriage Total', 'Single Parent Total', 'Civilian Married Total', 'Grand Total','Total Male', 'Total Female'])
enlisted_df3 = enlisted_df.drop(columns=['Single Male', 'Single Female', 'Joint Service Marriage Male', 'Joint Service Marriage Female','Single Parent Male','Single Parent Female', 'Civilian Married Male','Civilian Married Female', 'Grand Total','Total Male', 'Total Female'])
x = enlisted_df3['Pay Grade']
y = enlisted_df3['Single Total']
fig, ax = plt.subplots(figsize=(16,8)) 
ax.plot(x, enlisted_df3['Single Total'], label = 'Single Total') 
ax.plot(x, enlisted_df3['Single Parent Total'],label = 'Single Parent Total')
ax.plot(x, enlisted_df3['Joint Service Marriage Total'],label = 'Joint Service Marriage Total')
ax.plot(x, enlisted_df3['Civilian Married Total'],label = 'Civilian Marriage Total')
plt.legend(loc="upper right")
plt.title('Enlisted Service Members by Marital Status')
plt.xlabel('Enlisted Pay Grade')
plt.ylabel('Number of Service Members')
plt.rcParams.update({'font.size': 18})
plt.grid()
plt.show()

officer_df = ad_marital_status[10:20]
officer_df3 = officer_df.drop(columns=['Single Male', 'Single Female', 'Joint Service Marriage Male', 'Joint Service Marriage Female','Single Parent Male','Single Parent Female', 'Civilian Married Male','Civilian Married Female', 'Grand Total','Total Male', 'Total Female'])
x = officer_df3['Pay Grade']
fig, ax = plt.subplots(figsize=(16,8)) 
ax.plot(x, officer_df3['Single Total'], label = 'Single Total') 
ax.plot(x, officer_df3['Single Parent Total'],label = 'Single Parent Total')
ax.plot(x, officer_df3['Joint Service Marriage Total'],label = 'Joint Service Marriage Total')
ax.plot(x, officer_df3['Civilian Married Total'],label = 'Civilian Marriage Total')
plt.legend(loc="upper right")
plt.title('Officer Service Members by Marital Status')
plt.xlabel('Officer Pay Grade')
plt.ylabel('Number of Service Members')
plt.rcParams.update({'font.size': 18})
plt.grid()
