#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns # for the visualization


# In[2]:


#import dataset
trainDf = pd.read_csv("in_order_products__train.csv") #
orderDf = pd.read_csv("in_orders.csv")
depDf = pd.read_csv("in_departments.csv")
aisleDf = pd.read_csv("in_aisles.csv")
productDf = pd.read_csv("in_products.csv")
trainDf.head()


orderDf.head()


depDf.head()


aisleDf.head(


productDf.head()

# Data Preprocessing
dataset = pd.read_csv('product_transaction.csv', header = None)
transactions = []
for i in range(0, 7543):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

len(transactions)

dataset.head()

#transactions[:5]
depDf.head(10)




#load data into pandas dataframe..
big = pd.read_csv('big_baskt.csv', encoding="ISO-8859-1")
#to decode we use encode 


# In[14]:


big.head()


# In[15]:


big.nunique()


# In[16]:


#information of dataset..
big.info()
len(big)


# In[17]:


#Country with high count must be taken for testing purpose... can we divide based on location or similar state
big.Country.value_counts().head(5)


# In[18]:


big = big[big.Country == 'United Kingdom'] #we remove all dataset and work on a common place 


# In[19]:


len(big)           #remaining dataset which contain onlu "United Kingdom"


# In[20]:


big.Quantity.describe() #quantity for the UK


# In[21]:


#Quantity can not be negative so remove negative values..  and here is min value is -80995
big = big[big['Quantity']>0]
big.Quantity.describe()


# In[22]:


big = big[big['UnitPrice']>0]   # unit price shoud be greater than 0.
big.UnitPrice.describe()


# In[23]:


big.isnull().sum()


# In[24]:


big.dropna(subset=['CustomerID'],how='all',inplace=True) # we drop the all null values from th e customer 


# In[25]:


len(big)


# In[26]:


big.isnull().sum()


# In[27]:


#First date  with time available in our dataset
big['InvoiceDate'].max()


# In[28]:


# Last date with time  available in our datast
big['InvoiceDate'].min()


# In[29]:


#use latest date in our data as current date..

import datetime as dt
now = dt.date(2011,12,9) # a constant date to compare purchasing dates
#big.head()


# In[30]:


big['date'] = pd.DatetimeIndex(big.InvoiceDate).date # to seperate the date fromm the invoicedate


# In[31]:


big.head(2)


# In[32]:


dd=pd.DataFrame([big['InvoiceNo'],big['Description']]) # we frame a  new dataframeto identify the products acording to invoice


# In[33]:


dd.T.head()


# In[34]:


len(big)


# In[35]:


big['InvoiceNo'].nunique()            # no. of products listed in dataset


# In[ ]:





# In[ ]:





# In[36]:


#group by customer by last date they purchased...

recency_df = big.groupby(['CustomerID'],as_index=False)['date'].max()
recency_df.columns = ['CustomerID','LastPurchaseDate']
recency_df.head()


# In[37]:


#calculate how often he is purchasing with reference to latest date in days..
#                       "Recency"
recency_df['Recency'] = recency_df.LastPurchaseDate.apply(lambda x : (now - x).days)


# In[38]:


recency_df.head()


# In[39]:


recency_df.drop(columns=['LastPurchaseDate'],inplace=True)


# In[40]:


#check frequency of customer means how many transaction has been done..
#                          "Frequency"
frequency_df = big.copy()
frequency_df.drop_duplicates(subset=['CustomerID','InvoiceNo'], keep="first", inplace=True) 
frequency_df = frequency_df.groupby('CustomerID',as_index=False)['InvoiceNo'].count()
frequency_df.columns = ['CustomerID','Frequency']
frequency_df.head()


 

#calculate how much a customer spend in the each transaction...
#                      Expenditure
big['Total_cost'] = big['UnitPrice'] * big['Quantity']





#check summed up spend of a customer with respect to latest date..

Expenditure_df=big.groupby('CustomerID',as_index=False)['Total_cost'].sum()
Expenditure_df.columns = ['CustomerID','Expenditure']





Expenditure_df.head()


# In[44]:


Expenditure_df.head() #without index= false we dont have expenditure





#Combine all together all dataframe in so we have recency, frequency and monetary values together..

#combine first recency and frequency..
rf = recency_df.merge(frequency_df,left_on='CustomerID',right_on='CustomerID')

#combibe rf frame with monetary values..

rfe = rf.merge(Expenditure_df,left_on='CustomerID',right_on='CustomerID')

rfe.set_index('CustomerID',inplace=True)


# In[46]:


rfe.head()


# In[47]:


#checking correctness of output..
big[big.CustomerID == 12346.0]


# In[48]:


(now - dt.date(2011,1,18)).days == 325 # to check the recency that he is coming or not to shop

#bring all the quartile value("Recency","Frequency","Expenditure") in a single dataframe

rfe_segmentation = rfe.copy()


# In[50]:


from sklearn.cluster import KMeans
# get right number of cluster for K-means so we neeed to loop from 1 to 20 number of cluster and check score.
#Elbow method is used to represnt that. 
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(rfe_segmentation).score(rfe_segmentation) for i in range(len(kmeans))]
plt.plot(score,Nc)
plt.ylabel('Number of Clusters')
plt.xlabel('Score')
plt.title('Elbow Curve')
plt.show()


# In[51]:


#fitting data in Kmeans theorem.
kmeans = KMeans(n_clusters=3, random_state=0).fit(rfe_segmentation)


# In[52]:


# this creates a new column called cluster which has cluster number for each row respectively.
rfe_segmentation['cluster'] = kmeans.labels_


# In[53]:


#check our hypothesis

rfe_segmentation[rfe_segmentation.cluster == 0].head(5)


# In[54]:


'''
cluster 0 have high recency rate which is bad. cluster 1 and cluster 2 having low so they are in race of platinum
and gold customer.
'''
sns.boxplot(rfe_segmentation.cluster,rfe_segmentation.Recency)


# '''
# cluster 0 have high recency rate which is bad. cluster 1 and cluster 2 having low so they are in race of platinum
# and gold customer.
# '''

# In[55]:


sns.boxplot(rfe_segmentation.cluster,rfe_segmentation.Frequency)


# cluster 0 have low frequency rate which is bad. cluster 1 and cluster 2 having high so they are in 
# race of platinum and gold customer.
# 



sns.boxplot(rfe_segmentation.cluster,rfe_segmentation.Expenditure)


# Based on customer Segmentation we found out cluster 1 is Platinum customers Cluster 2 is Gold Customers Cluster 3 is Silver Customers



# Plaatanium                                                                 #.5(platanium,gold,silver)
print(len(rfe_segmentation[rfe_segmentation.cluster == 1]))
rfe_segmentation[rfe_segmentation.cluster == 1].head()



#gold
print(len(rfe_segmentation[rfe_segmentation.cluster == 2]))
rfe_segmentation[rfe_segmentation.cluster == 2].head()


#silver
print(len(rfe_segmentation[rfe_segmentation.cluster == 0]))
rfe_segmentation[rfe_segmentation.cluster == 0].head()


#

#import dataset
#trainDf = pd.read_csv("in_order_products__train.csv")
#orderDf = pd.read_csv("in_orders.csv")
#depDf = pd.read_csv("in_departments.csv")
#aisleDf = pd.read_csv("in_aisles.csv")
#productDf = pd.read_csv("in_products.csv")


# In[61]:


trainDf.head()


# In[62]:


orderDf.nunique()   # order_dow = day of week


# In[ ]:





# In[63]:


len(orderDf)


# In[64]:


depDf.head()


# In[65]:


aisleDf.head()


# In[66]:


productDf.head()


# In[67]:


orderDf.shape                                                                      #.1 (tottal no. of transactions)


# this is the dataset of 34,21,083 transactions only.......

# In[68]:


productDf.shape


# here, only 49,688 products are listed,which are ready to sell.......


#get distribution of number of order in percent
print("                  Get distribution of number of order in percent")
sns.set_style('whitegrid') # color of graph
customerNumOrderFrame = orderDf.groupby("user_id",as_index = False)["order_number"].max()
num_bins = 30      # no of plots in graph,  # alpha = color intensity #no of grids 
plt.hist(customerNumOrderFrame["order_number"] , num_bins, density=1, color='red', alpha=1)
plt.title("Get distribution of number of order in percent")
plt.ylabel('Percent', fontsize=13)
plt.xlabel('number of order', fontsize=13)
plt.savefig("hist5.png")


# In[ ]:





# In[70]:


priorDf = pd.read_csv("in_order_products__prior.csv")
trainDf = trainDf.append(priorDf,ignore_index = True)
#Now a product count data frame can be created by counting the order_id for each product_id
productCountDf = trainDf.groupby("product_id",as_index = False)["order_id"].count()


# In[71]:


productCountDf.head()


# In[72]:


priorDf.head(3)


# In[73]:


trainDf.head()


# In[74]:


#Top 100 most frequently purchased products
topLev = 100

#Here order_id is the count so we need to sort the data frame w.r.t order_id
productCountDf = productCountDf.sort_values("order_id",ascending = False)

topProdFrame = productCountDf.iloc[0:topLev,:]
topProdFrame = topProdFrame.merge(productDf,on = "product_id")
productId= topProdFrame.loc[:,["product_id"]]


# In[75]:


topProdFrame.head() 
#trainDf.head(10)#.6 top selling products


# In[76]:


# create a bar chart, rank by value
def Number_of_sales():
    a=trainDf.product_id.value_counts()[:5].plot(kind="bar", title="Total Number of Sales by Product").set(xlabel="Item", ylabel="Total Number")
    #plt.savefig("N_o_s.png")
    return a
Number_of_sales()


# In[77]:


#a.legend(["AAA","b","g","y","yy"])
#Number_of_sales()


# In[78]:


display(topProdFrame.loc[:,["product_name","product_id"]].head()) 


# In[79]:


# plot time series chart of number of items by day
#productDf["product_name"].value_counts().plot(title="Total Number of Items Sold by Date").set(xlabel="Date", ylabel="Total Number of Items Sold")




#big.sort_index(inplace=True)
#big['UnitPrice'].plot()

#big["UnitPrice"].plot(kind='pie')





#Now
#orderDf['order_id'].plot()




# create a pie chart, by %aisle(product row)
aisleDf.aisle.value_counts()[:10].plot(kind="pie", title="DIfferent Aisle for Shopping")
#plt.savefig("aisle.png")


# In[83]:


#Now
orderDf.head()







# no of ordrs in a week ...in givedd Daata


grouped = orderDf.groupby("order_id")["order_dow"].aggregate("sum").reset_index(name='order_dow')
grouped = grouped.order_dow.value_counts()                                              #.3(days graph)


print("          Number of Order Per Week")
sns.barplot(grouped.index, grouped.values)#.3
plt.title("Number of Order Per Week")
plt.ylabel('Number of orders', fontsize=13)
plt.xlabel('Number of Days in a week', fontsize=13)
plt.savefig("N_o_O_p_w.png")
plt.show()
#print("shows a graph")


print('Sunday and Monday is the busiest day of the week with the highest sales while Thursday is the quietest day with the lowest sales . This is an interesting insight, the owner of the Bakery should launch some promotion activities to boost up sales in the middle of the week when sales are slowest.')



grouped1=orderDf.groupby('order_id')['order_hour_of_day'].aggregate("sum").reset_index()
grouped1= grouped1.order_hour_of_day.value_counts()    

print("          Number of Order per Hour ")
sns.barplot(grouped1.index, grouped1.values)        #.4
plt.title(" Number of Order per Hour")
plt.ylabel('Number of orders', fontsize=13)
plt.xlabel('Order in a day(In terms of Hours)', fontsize=13)

plt.show()#.4(hourly graph)



# no of unique custmer in the  whole dataset


# In[89]:


len(set(orderDf.user_id))                                                                #.2 (no of users)


# In[ ]:





# In[90]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules



# now we merge the train and order dataset to get the real product selling dataset


# In[92]:


trainDf = trainDf.append(priorDf,ignore_index = True)


# In[93]:


trainDf.head()


# In[94]:


#For counting each product, we can assign reordered column as 1




trainDf['reordered'] = 1


productCountDf = trainDf.groupby("product_id",as_index = False)["order_id"].count()


newproductCountDf=productCountDf.merge(productDf, left_on='product_id', right_on='product_id', how='inner') 
newDf = newproductCountDf[['product_id','product_name']]
newDf.head()


newproductCountDf.head()



len(productId)


# Now we will filter the orders and get orders containting the the most frequently purchased products

# In[100]:


df = trainDf[0:0]
for i in range(0,99):
    pId = productId.iloc[i]['product_id'] 
    stDf = trainDf[trainDf.product_id == pId ]
    df = df.append(stDf,ignore_index = False)



df=df.reset_index()



# Now we need to consolidate the items into 1 transaction per row with each product 1 hot encoded. Each row will represent an order and each column will represent product_id. If the cell value is '1' say (i,j) then ith order contains jth product.

# In[102]:


basket = df.groupby(['order_id', 'product_id'])['reordered'].sum().unstack().reset_index().fillna(0).set_index('order_id')


# In[103]:


# Convert the units to 1 hot encoded values
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1 :
        return 1





basket_sets = basket.applymap(encode_units)





#basket_sets.head()
basket_sets.head()

basket_sets.size


# Now that the data is structured properly, we can generate frequent item sets that have a support of at least 1%

# In[ ]:


# Build up the frequent items
requent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)


# In[93]:


type(requent_itemsets)


# In[95]:


# Create the rules
rules = association_rules(requent_itemsets, metric="lift", min_threshold=1)
rules.head()




# # information  we will provide to client

# In[157]:


orderDf.shape                               #.1


# # you have 34,21,083 no. of transaction till now!!!!!


len(set(orderDf.user_id))                 #.2






# sales groupby weekday
#bread_groupby_weekday = bread.groupby("Weekday").agg({"Item": lambda item: item.count()})
#bread_groupby_weekday


# top selling products
# 




topProdFrame.head(int(input("enter the no.for top selling product list = "))) #.6




# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
print(type(rules))
results = list(rules)

# Total number of rules generated
print(len(results))




cnt=0
for item in results:
    pair = item[0]
    items = [x for x in pair]
    print("Rule :"+ str(items[0]) + "->" + str(items[1]) )
    print("Support : {}".format(item[1]))
    print("Confidence : {}".format(item[2][0][2]))
    print("List : {}".format(item[2][0][3]))
    print("\n-------------------------------------------------\n")
    cnt = cnt + 1
    if cnt >2:
        break;

qq=pd.DataFrame()




product1=[]
product2=[]
Support1=[]
Confidence =[]
Lift1 = []

cnt=0
for item in results:
    pair = item[0]
    items = [x for x in pair]
    product1.append(str(items[0]))
    product2.append(str(items[1]))
    Support1.append(round(item[1],3))
    Confidence.append(round(item[2][0][2],3))
    Lift1.append(round(item[2][0][3],3))
    #print("\n-------------------------------------------------\n")
    cnt = cnt + 1
    if cnt >10:
        break;



qq['product1']=product1
qq['product2']=product2
qq['Support']=Support1
qq['Confidence']=Confidence
qq['Lift']=Lift1



qq.sort_values(['Lift'],ascending=False,inplace=True)


# In[173]:


qq=qq.head(int(input("enter the no.. for top related product")))


# In[170]:


# plot time series chart of number of items by month
#priorDf["product_name"].resample("M").count().plot(figsize=(12,5), grid=True, title="Total Number by Items Sold by Month").set(xlabel="Date", ylabel="Total Number of Items Sold")






