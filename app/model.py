import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LassoCV

filepath = "/Users/sampark/Documents/Work/Kaggle/housePrices/"
filename_train = "train.csv"
filename_test = "test.csv"

train = pd.read_csv(filepath + filename_train)
test = pd.read_csv(filepath + filename_test)

data = pd.concat([train, test], keys=('x', 'y'))
data = data.drop(["Id"], axis = 1)

num_data = data._get_numeric_data().columns.tolist()

cat_data = set(data.columns) - set(num_data)

for col in num_data:
    data[col].fillna(data[col].mean(), inplace=True)
    
for col in cat_data:
    data[col].fillna(data[col].mode()[0], inplace=True)
    
data['SalePrice'] = np.log1p(data['SalePrice'])

def mod_outliers(data):
    df1 = data.copy()
    data = data[["LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtUnfSF", "TotalBsmtSF",
                "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea", "WoodDeckSF",
                "OpenPorchSF"]]
    
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    
    iqr = q3 - q1
    
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    for col in data.columns:
        for i in range(0, len(data[col])):
            if data[col][i] < lower_bound[col]:
                data[col][i] = lower_bound[col]
                
            if data[col][i] > upper_bound[col]:
                data[col][i] = upper_bound[col]
                
    for col in data.columns:
        df1[col] = data[col]
        
    return(df1)

data = mod_outliers(data)

data = data[["OverallQual", "GarageCars", "YearBuilt", "SalePrice"]]

train = data.loc["x"]
test = data.loc["y"]
test = test.drop(["SalePrice"], axis = 1)

y = train["SalePrice"]  
train_x = train.drop(["SalePrice"], axis = 1)
test_x = test

lasso = LassoCV(alphas = [1, 0.1, 0.01, 0.001, 0.0001]).fit(train_x, y)

pickle.dump(lasso, open('lasso.pkl', 'wb'))

loaded_lasso = pickle.load(open('lasso.pkl', 'rb'))

print(np.expm1(loaded_lasso.predict([[5, 1, 1961]])))



