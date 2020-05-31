import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.dummy import DummyClassifier


#data = pd.read_excel("Temp.xlsx")
data = pd.read_excel("Temp_clean_FeSel.xlsx")

print(data)

# for column in data.columns:
#         #data[column].str.join(sep='*').str.get_dummies(sep='*')
#         if data[column].dtype==object:
        
#             dummyCols=pd.get_dummies(data[column],prefix=column)
#             del data[column]
#             data=data.join(dummyCols)
data_clean=pd.DataFrame()
for column in data.columns:

    col=data[column].str.get_dummies(sep=',').add_prefix(column+"_")
    print(col)
    data_clean=pd.concat([data_clean,col],axis=1)
#pd.get_dummies(cleaned, prefix='g').groupby(level=0).sum()

print(data_clean)
#data_clean.to_excel("output.xlsx")

    
from sklearn.cluster import DBSCAN

clustering = KMeans(n_clusters=5).fit_predict(data_clean)
print(clustering)


import matplotlib.pyplot as plt  
from matplotlib import style
# cost=[]
# for i in range(1, 100): 
#     KM = KMeans(n_clusters = i) 
#     KM.fit(data_clean) 
      
#     # calculates squared error 
#     # for the clustered points 
#     cost.append(KM.inertia_)      
  
# plot the cost against K values 
# plt.plot(range(1, 100), cost, color ='g', linewidth ='3') 
# plt.xlabel("Value of K") 
# plt.ylabel("Sqaured Error (Cost)") 
# plt.show() # clear the plot 



from kmodes.kmodes import KModes

# # random categorical data
# #data = np.random.choice(20, (100, 10))

# km = KModes(n_clusters=3, init='Huang', n_init=5, verbose=1)

# clusters = km.fit_predict(data_clean)

# # Print the cluster centroids
# print(km.cluster_centroids_)

cost = []
for num_clusters in list(range(1,100)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(data)
    cost.append(kmode.cost_)
plt.plot(range(1, 100), cost, color ='g', linewidth ='3') 
plt.xlabel("Value of K") 
plt.ylabel("Squared Error (Cost)") 
plt.show()

kmode = KModes(n_clusters=16, init = "Cao", n_init = 1, verbose=1).fit_predict(data)
#Number of cluster n-th
n_th=5
print(kmode[n_th])
