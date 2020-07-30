# 使用KMeans进行聚类,分为了四类，输出文件中每款车型的reasult为分类结果
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd

data=pd.read_csv('CarPrice_Assignment.csv')
data_new=data.drop(["CarName","car_ID","citympg","highwaympg"],axis=1)
le=preprocessing.LabelEncoder()
columns=['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']
for column in columns:
    data_new[column]=le.fit_transform(data_new[column])
min_max_scaler=preprocessing.MinMaxScaler()
data_new=min_max_scaler.fit_transform(data_new)
train_new=PCA(n_components=4).fit_transform(data_new)
kmeans = KMeans(n_clusters=4)
kmeans.fit(train_new)
predict_y = kmeans.predict(train_new)
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:u'result'},axis=1,inplace=True)
result.to_csv("car_classify.csv",index=False)
