import pandas as pd
X = pd.read_csv('test.csv',index_col='Id')
y=pd.read_csv('gt.csv',index_col='Id')

y=y[['SaleStatus']]


data=pd.concat([X,y],axis=1)
print(data.head)
data.dropna(inplace=True)

y=data[data.columns[31]]

X=data.drop('SaleStatus',axis=1)
df=pd.DataFrame(X)

X_object=df.select_dtypes(include=['object']) #containing all string values 
X_numeric=df.select_dtypes(exclude=['object']) #containing all numeric values
print(X_object)


df=X_numeric[['MSSubClass','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','1stFlrSF','2ndFlrSF','GrLivArea','GarageYrBlt','GarageArea']]

df1=X_numeric[['OverallQual','OverallCond','LowQualFinSF','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','GarageCars']]
df_norm = (df - df.mean()) / (df.max() - df.min())

df_norm=pd.concat([df_norm,df1],axis=1)
X_numeric=df_norm

X_numeric=X_numeric.drop(['OverallCond','BsmtHalfBath','LowQualFinSF','YearBuilt','KitchenAbvGr','2ndFlrSF'],axis=1)

X_object=X_object.drop(['Condition1','Condition2'],axis=1)

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
for col in X_object.columns:
    EncData=X_object[col]
    enc.fit(EncData.values)
    X_object[col]=enc.transform(X_object[col])

X=pd.concat([X_numeric,X_object],axis=1) #final pre-processed values
print("##################")
print(X)
from sklearn.externals import joblib
loaded_model = joblib.load('model_Randomforest.pkl')

result = ('RandomForestAccuracy2:{:.3f}'.format(loaded_model.score(X,y)))
print(result)


y_true,y_predicted=y,loaded_model.predict(X)


print(y_predicted)

df=pd.DataFrame(y)
filename = 'out.csv'
df.to_csv(filename,index=False)
