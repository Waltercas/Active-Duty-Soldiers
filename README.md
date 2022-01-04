# Active Duty Soldiers

## Background

The US Military is a constantly evolving organization with changing ideals. The Army's view in recent history is one that is focused on individuals.  Forget technology, vehicles, aircraft and weapons, Soldiers are the Army's most valuable asset. Without individuals that are willing to serve all the expensive pieces of equipment are left with nobody to man them. 

Soldiers have different needs and there are different factors that determine those differences. Factors that determine soldier's needs include age, gender and marital status to name a few. In this project I will be looking specifically at marital status in service members of all branches.  Marital status is important to know as leaders to better accommodate both single and married service members. Single and married soldiers have different needs but its important to fulfill those needs for both equally. Married soldiers require services like childcare, family counseling and other services that single soldiers don't. Single soldiers require different services like recreation and barracks housing.    



## Data

The data frame that I used for this project was sourced from Data.gov created in 2010.  In the data frame are active duty service members sorted by rank and marital status.  There was no missing data or nan's in the data frame.  The data frame encompasses almost 1.5 million service members from all branches.


![Screenshot from 2021-12-15 20-09-29](https://user-images.githubusercontent.com/92833923/148118243-46ffa108-f6cd-4174-95b7-43bd9e97f039.png)



All columns are numerical except the 'Pay Grade' column.                                                                                                

The rows are indexed by pay grade ranging from E-1 to O-10 

## Data Exploring

When I started exploring the data I wanted to look at the relationship of rank and marital status so I plotted  both officers and enlisted in two different subplots.

![Marrital Status Multiple Plot](https://user-images.githubusercontent.com/92833923/148117743-2b72484c-6c93-435d-805b-cc4335497ad0.png)

This graph showed me some interesting facts. 

First I can see that at the rank E-3 there is the most amount of single soldiers. This makes sense to me since most of soldiers in this rank are young. At E-5 you can see the most amount married on average from my experience in the Army usually E-5s have been in about 5 years so one can deduce they would be aged around 23 years old. 

After this graph I figured it would be best for me to look at marital status as a whole so I plotted two separate bar charts of the average number of soldiers by marital status by rank.
![Marital Status Mean Enlisted](https://user-images.githubusercontent.com/92833923/148117784-74a4583d-b69a-4134-a4d7-f39fc412add0.png)

![Marital Status Mean By Officer](https://user-images.githubusercontent.com/92833923/148117836-b15b0fe1-88ce-4f5a-bbac-aa9128bf9ade.png)



I decided it would be easier to just look at both together so I created a pie chart of all active duty service members by marital status.

![AD Pie Chart](https://user-images.githubusercontent.com/92833923/148118345-23d190a4-7e26-490d-ba40-4d2035d8cda5.png)

Now after creating this pie chart it looks like married personnel out number single service members so then I transitioned to hypothesis testing

## Hypothesis Tests

I decided a two sample z-test would be the best test to use I could determine if the difference between two populations is significantly different from one another. 

### Married vs Unmarried

 I started by setting my alpha to 0.05 just following a general standard. Then I stated the null and alternate hypothesis.

Null hypothesis = The population of married soldiers is statistically significantly different from the population of unmarried soldiers.

Alternative hypothesis = The population of married soldiers is not statistically significantly different from the population of unmarried soldiers.

The results of the z test is 0.1

P value is 0.9

Since the P value is greater than our alpha threshold we fail to reject the null hypothesis. This means that we found that there is no significant difference between married and unmarried service members.

### Married Enlisted vs Married Officers

For this test I also set the alpha to 0.05 to keep all tests the same.

Null hypothesis and Alternative hypothesis will be the same for this test as well.

The results of this z test is -1.2.

P value is 0.2

Again the P value is greater than the alpha here we also fail to reject the null hypothesis. Therefore there is no significant difference between the number of married enlisted vs married enlisted.

### Single Enlisted vs Single Officers

Finally alpha stays constant as well as Null hypothesis and Alternative hypothesis.

The results of this z test is -0.6.

P value is 0.5

P value is greater than alpha of 0.05 so we reject the null hypothesis meaning there is no significant difference between number of single enlisted and single officers.



## Conclusion

According to our tests we can conclude that since there is no significant difference between service members that are married and unmarried in all branches we can say that we as leaders should look to spread services and resources equally.

### Next Steps

In the future I would like to look at better data in general. The data is acquired is from 2010 so I would like to use more recent data. Also I would like to look at more detailed data, more specifically I would like to see demographics like race age and sex.

 

