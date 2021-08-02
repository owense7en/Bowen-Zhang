from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame

#cancel the Scientific notation
pd.set_option('display.float_format',lambda x : '%.2f' % x)

#Retieve the dataset
filename_train = "house-prices-advanced-regression-techniques/train.csv"
filenam_test = "house-prices-advanced-regression-techniques/test.csv"
data_train = pd.read_csv(filename_train,header = 0,sep=',')
data_test = pd.read_csv(filenam_test,header = 0,sep=',')

#data_train.info()
#data_train.describe().T

#Get the categorical dataset
categorical_feature = []
threshold = 20
for each in data_train.columns:
    if data_train[each].nunique() < threshold:
        categorical_feature.append(each)

#Get the numerical dataset
numerical_feature = []
for each in data_train.columns:
    if each not in categorical_feature:
        numerical_feature.append(each)

print("categorical_feature:",categorical_feature)
print("numerical_feature", numerical_feature)
print("categorical_feature length:",len(categorical_feature))
print("numerical_feature length", len(numerical_feature))

#reduce the outliers
outlier_indexes = []
for i in [col for col in data_train.columns if data_train[col].dtype != 'O']:
    if i != 'id':
        outlier = []
        Q1 = data_train[i].quantile(0.25)
        Q3 = data_train[i].quantile(0.75)
        IQR = Q3 - Q1
        lower_outlier_finder = Q1 - (1.5 * IQR)
        higher_ouliier_finder = Q3 + (1.5 * IQR)

        threshold = 3
        for j in data_train[i]:
            z = (j - data_train[i].mean()) / data_train[i].std()
            if z > threshold:
                outlier.append(j)
                index = data_train[data_train[i] == j].index[0]
                outlier_indexes.append(index)

data_train = data_train.drop(outlier_indexes, axis=0).reset_index(drop=True)
all_data = pd.concat([data_train,data_test],axis=0,sort=False)

#Find how many nan in each column
pd.set_option('display.max_rows', 100)
info_count = pd.DataFrame(all_data.isnull().sum(),columns=['Count of NaN'])
dtype = pd.DataFrame(all_data.dtypes,columns=['DataTypes'])
info = pd.concat([info_count,dtype],axis=1)


#Fill the nan value
all_data['LotFrontage'].interpolate(method='linear',inplace=True)

for i in info.T:
    if i == "id" or i == "SalePrice" or i == "LotFrontage":
        continue
    else:
        if (info.T[i][0] == 0):
            continue
        elif (info.T[i][0] < 400):
            all_data[i].fillna(all_data[i].value_counts().index[0],inplace=True)
        else:
            lableEnc = LabelEncoder()
            lableEnc.fit(list(all_data[i].values))
            all_data[i] = lableEnc.transform(list(all_data[i].values))

#Get the category dataset and put them to the 0,1 dataset.
list_ = ["MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig",
        "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
        "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
        "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual",
        "Functional", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "SaleType",
        "SaleCondition"]

for feature in list_:
    all_data[feature] = all_data[feature].astype("category")
    all_data = pd.get_dummies(all_data, columns=[feature])


#Slipt the data to test_data and train_data
sum = 0
for index,number in all_data["SalePrice"].iteritems():
    if str(number) !=  "nan":
        sum = sum+1
sum_row = 0
for index,number in all_data["SalePrice"].iteritems():
    sum_row = sum_row+1

train_data = all_data[0:sum]
test_data = all_data[sum:sum_row]

test_data = test_data.drop("SalePrice",axis=1)
test_data = test_data.drop("Id",axis=1)

y = train_data["SalePrice"]
X = train_data.drop(["Id","SalePrice"],axis=1)

print("target class:",len(y.unique()))

x_train,x_test, y_train, y_test = train_test_split(X,y,test_size=0.1)





#Linear Regression Model
model = LinearRegression()
model.fit(x_train,y_train)
predict_price = model.predict(x_test)

residuals = (predict_price - y_test)/y_test
average = np.mean(residuals)

plt.hist(residuals)
plt.show()





#XGBoost Model1: for the original regression
import xgboost as xgb
from xgboost import plot_importance
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 999999,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 0,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
    'verbosity':0
}

plst = list(params.items())

dtain = xgb.DMatrix(x_train,y_train)

num_rounds = 500

model = xgb.train(plst,dtain,num_rounds)

dtest = xgb.DMatrix(x_test)
predict_price_XGBoost = model.predict(dtest)

print(predict_price_XGBoost)






#XGBoost Model1: for the Scikit-learn
model = xgb.XGBRegressor(max_depth=5,
                         learning_rate=0.1,
                         n_estimators=160,
                         silent=True,
                         objective='reg:gamma',
                         verbosity=0)

model.fit(x_train, y_train)

predict_price_XGBoost2 = model.predict(x_test)

print(predict_price_XGBoost2)

plot_importance(model)
plt.show()







#LightGBM Model
import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(x_train,y_train)

predict_price_LightGBM = clf.predict(x_test)
print(predict_price_LightGBM)








#Catboost Model
import catboost as catb
catboost_model = catb.CatBoostClassifier(iterations = 2,
                                         learning_rate = 1,
                                         depth =2)

catboost_model.fit(x_train,y_train)
catboost_predict = catboost_model.predict(x_test)

print(catboost_predict)







#tersorflow Model
import tensorflow as tf
import tensorflow.keras.models as load_model
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(999999, activation='softmax'))

model.compile(optimizer=tf.optimizers.Adam(0.01),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

model.fit(x_train, y_train, epochs=5, batch_size=32)
predict_tesorflow_model = model.predict(x_test)
print(predict_tesorflow_model)






#pytorch 
import torch
import torch.nn as nn

numeric_x_columns = list(numerical_feature)
numeric_x_columns.remove('SalePrice')
numeric_x_columns.remove('Neighborhood')
numeric_y_columns = ['SalePrice']

numeric_x_df = DataFrame(data_train[numerical_feature], columns=numeric_x_columns)
numeric_y_df = DataFrame(data_train[numerical_feature], columns=numeric_y_columns)

numeric_x_df= numeric_x_df.astype(float)
numeric_y_df= numeric_y_df.astype(float)


numeric_x = torch.tensor(numeric_x_df.values, dtype=torch.float)
numeric_y = torch.tensor(numeric_y_df.values, dtype=torch.float)

class Net(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)

    def forward(self, x):
        y_pred = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(y_pred).clamp(min=0)
        y_pred = self.linear3(y_pred).clamp(min=0)
        y_pred = self.linear4(y_pred)
        return y_pred

H1, H2, H3 = 500, 1000, 200
D_in, D_out = numeric_x.shape[1], numeric_y.shape[1]

model1 = Net(D_in, H1, H2, H3, D_out)

criterion = nn.MSELoss(reduction='sum')

optimizer = torch.optim.SGD(model1.parameters(), lr=1e-4)

losses1 = []

for t in range(500):
    y_pred = model1(numeric_x)

    loss = criterion(y_pred, numeric_y)
    print(t, loss.item())
    losses1.append(loss.item())

    if torch.isnan(loss):
        break

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#Get the result of mean_squared for each model
mean_squared = mean_squared_error(y_test,predict_price)
print(mean_squared)


