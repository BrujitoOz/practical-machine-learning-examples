#%% dataset
import numpy as np
import pandas as pd
dataset = pd.read_csv("../Social_Network_ads.csv")
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values
#%% split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
#%% scale
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
#%% train
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", random_state = 0)
classifier.fit(x_train, y_train)
#%% prediction
y_pred = classifier.predict(x_test)
print(y_pred)
print(y_test)
#%% Matrix
from sklearn .metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
def MatrizdeConfusion(cm):
    print(pd.DataFrame({' ':['No', 'Si'],'No':cm[0],'Si':cm[1]}))
    aciertos = 0
    desaceritos = 0
    for i in range(len(cm[0])):
        aciertos += cm[i][i]
    desaceritos = sum(sum(cm))
    desaceritos -= abs(aciertos) 
    print("Cantidad aciertos:",aciertos)
    print("Cantidad desaciertos:",desaceritos)
MatrizdeConfusion(cm)
#%% cross
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X= x_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()
#%% grid search
from sklearn.model_selection import GridSearchCV
parameters = [
    {'C': [1,10,100,1000],
    'kernel':['Linear']},
    {'C': [1,10,100,1000],
    'kernel': ['rbf'], 'gamma':[0.5,0.1,0.01, 0.0001]}
]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_ 
best_params = grid_search.best_params_
#%% view
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ['red', 'green'][i], label = j)
plt.title('Clasificador SVM (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()
#%%
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ['red', 'green'][i], label = j)
plt.title('Clasificador SVM (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()