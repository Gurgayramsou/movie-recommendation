'''
here i have used k-means clustoring algorith to clustor
movies based on similarity
then used logistic regression to relate the movies and its clustored class
after that taking input from user and find which clustor it belongs
based on that movies bein '''
#importing libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import random

#importing dataset
x=pd.read_csv('u1.csv',sep='\t')
itemid=x.iloc[:,[1]].values # independent variable for log_reg



# getting items from u.item file
with open('u.item',"r",encoding='latin-1') as fhand:
    movies=[]
    for line in fhand.readlines():
        a=line.split('|')
        movies.append(a[1])


#dataet x  to standard scaling because we used eucledian distace as parameter
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x=ss.fit_transform(x)


#fitting kmeans to dataset
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=500)
y_kmeans=kmeans.fit_transform(x)
y_pred=kmeans.fit_predict(x)

#applying logistic regression to predict the class of all the set of data in x
from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression()
regressor.fit(itemid,y_pred)


#asking user for watched movie
movie=input('enter the movie recently watched: ')
print("---------------------------------------------------------")
idx=movies.index(movie)
cls_pred=regressor.predict([[idx]])
n=random.randint(0,len(y_pred))
count=0
while(True):
    n=random.randint(0,len(y_pred))
    if y_pred[n]==cls_pred:
        j=itemid[n]
        print(movies[j[0]])
        count+=1
    if count==10:
        break



    
    
    
    















