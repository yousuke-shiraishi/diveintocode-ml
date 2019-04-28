import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = np.log(df.loc[:,['GrLivArea','YearBuilt']].values)
y = np.log(df.loc[:,['SalePrice']].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)
regr = make_pipeline(
    StandardScaler(),
    LinearRegression())
reg = regr.fit(X_train, y_train)
y_pred = reg.predict(X_test)
plt.scatter(X_test[:,0],y_test,label ='GrLivArea' )
sort_idex = X_test[:,0].argsort() 
plt.plot(X_test[:,0][sort_idex],y_pred[sort_idex],color="red")
plt.ylabel("SalePriceの対数")
plt.show()

plt.scatter(X_test[:,1],y_test,label = 'YearBuilt')
sort_idex = X_test[:,1].argsort() 
plt.plot(X_test[:,1][sort_idex],y_pred[sort_idex], color = 'red')
plt.ylabel("SalePriceの対数")
plt.show()