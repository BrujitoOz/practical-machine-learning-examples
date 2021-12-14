#%% Read
import numpy as np
import pandas as pd
dataset = pd.read_csv("../Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
#%% Train
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state=0)
regression.fit(x,y)
#%% Pred
y_pred = regression.predict(x)
#%% Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MSE:", mean_squared_error(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
print("MSA:", mean_absolute_error(y, y_pred))
#%% Visual
import matplotlib.pyplot as plt
plt.scatter(x,y, color="red")
plt.plot(x, regression.predict(x))
plt.show()
# %%
