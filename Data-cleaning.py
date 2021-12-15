import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
 
ad_location = pd.read_excel('Data/DMDC_Website_Location_Report_2109.xlsx')
ad_rank = pd.read_excel('Data/AD_by_rank_2021.xlsx')
ad_location = ad_location.loc[:, ['ARMY']]
print(ad_location.head(10))

