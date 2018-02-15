
import pandas as pd
data = pd.read_csv('train.csv',index_col='Id')

print(data.head())
data.dropna(inplace=True)  #for dropping rows with NAN values
X=data.drop(data.columns[31],axis=1) #for removing label column and putting features in X
y=data[data.columns[31]]
print(y)



df=pd.DataFrame(X)

X_object=df.select_dtypes(include=['object']) #containing all string values 
X_numeric=df.select_dtypes(exclude=['object']) #containing all numeric values
print(X_object)


df=X_numeric[['MSSubClass','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','1stFlrSF','2ndFlrSF','GrLivArea','GarageYrBlt','GarageArea']]

df1=X_numeric[['OverallQual','OverallCond','LowQualFinSF','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','GarageCars']]
df_norm = (df - df.mean()) / (df.max() - df.min())

df_norm=pd.concat([df_norm,df1],axis=1)

import matplotlib.pyplot as plt
names=list(df_norm.columns[list(range(20))])

correlations= df_norm.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=list(range(20))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

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

print("final data")
print(X)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

param_grid = [ {'C': [100,400,800], 'kernel': ['linear','poly','rbf'], 'degree': [3,4,5], 'gamma': [0.1,0.01,0.001]}] 
                                                                # Parameter Grid
clf_set = GridSearchCV(SVC(), param_grid, cv = 5)
								# Classifier set 

clf_set.fit(X,y)		# The optimisation

print(clf_set.best_params_)


meanScore = clf_set.cv_results_['mean_test_score']
stddevScore = clf_set.cv_results_['std_test_score']

print()

for mean, stddev, params in zip(meanScore, stddevScore, clf_set.cv_results_['params']):
     print("%0.3f (+/-%0.03f) for %r" % (mean, stddev * 2, params))


#using Best Parameters
svc=SVC(C= 800,gamma= 0.001, degree= 3, kernel= 'rbf')
svc.fit(X,y)





from sklearn.ensemble import RandomForestClassifier
param_grid = {
    'bootstrap': [True,False],
    'max_depth': [5,6],
    'max_features': ['sqrt','auto'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [7,9],
    'n_estimators': [60,70,80]
}

from sklearn.grid_search import GridSearchCV
CV_rfc = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)
print(CV_rfc.best_params_)
print(CV_rfc.best_score_)

#using best parameters
forest=RandomForestClassifier(n_estimators=70,min_samples_split=7,max_features='auto',bootstrap=False,max_depth=6,min_samples_leaf=4,random_state=0)
forest.fit(X,y)






#output of trained model
from sklearn.externals import joblib
filename1 = 'model_Randomforest.pkl'
joblib.dump(forest, filename1)
filename2='model_SVC.pkl'
joblib.dump(svc,filename2)






#choosing Best Classifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
forest1=RandomForestClassifier(n_estimators=70,min_samples_split=7,max_features='auto',bootstrap=False,max_depth=6,min_samples_leaf=4,random_state=0)
forest1.fit(X_train,y_train)
svc1=SVC(C= 800,gamma= 0.001, degree= 3, kernel= 'rbf')
svc1.fit(X_train,y_train)
print("Accuracy of Random Forest ")
print('Accuracy:{:.3f}'.format(forest1.score(X_test,y_test)))
print("Accuracy of SVM")
print('Accuracy:{:.3f}'.format(svc1.score(X_test,y_test)))


