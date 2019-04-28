import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris= load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] =  iris.target_names[y]
pd.set_option('display.max_rows', None)
print("-------------------------------")
df[df['Species'].isin(['versicolor', 'virginica'])]

def decision_region(X_train, y_train, model, step=0.01, title='decision region', xlabel='xlabel', ylabel='ylabel', target_names=['versicolor', 'virginica']):
    """
    2値分類を2次元の特徴量で学習したモデルの決定領域を描く。
    背景の色が学習したモデルによる推定値から描画される。
    散布図の点は学習用データである。

    Parameters
    ----------------
    X_train : ndarray, shape(n_samples, 2)
        学習用データの特徴量
    y_train : ndarray, shape(n_samples,)
        学習用データの正解値
    model : object
        学習したモデルのインスンタスを入れる
    step : float, (default : 0.1)
        推定値を計算する間隔を設定する
    title : str
        グラフのタイトルの文章を与える
    xlabel, ylabel : str
        軸ラベルの文章を与える
    target_names= : list of str
        凡例の一覧を与える
    """
    # setting
    scatter_color = ['red', 'blue']
    contourf_color = ['pink', 'skyblue']
    n_class = 2

    # pred
    mesh_f0, mesh_f1  = np.meshgrid(np.arange(np.min(X_train[:,0])-0.5, np.max(X_train[:,0])+0.5, step), np.arange(np.min(X_train[:,1])-0.5, np.max(X_train[:,1])+0.5, step))
    mesh = np.c_[np.ravel(mesh_f0),np.ravel(mesh_f1)]
    pred = model.predict(mesh).reshape(mesh_f0.shape)

    # plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.contourf(mesh_f0, mesh_f1, pred, n_class-1, cmap=ListedColormap(contourf_color))
    plt.contour(mesh_f0, mesh_f1, pred, n_class-1, colors='y', linewidths=3, alpha=0.5)
    for i, target in enumerate(set(y_train)):
        plt.scatter(X_train[y_train==target][:, 0], X_train[y_train==target][:, 1], s=80, color=scatter_color[i], label=target_names[i], marker='o')
    patches = [mpatches.Patch(color=scatter_color[i], label=target_names[i]) for i in range(n_class)]
    plt.legend(handles=patches)
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
iris= load_iris()
X = iris.data
X = X[50:]
y = iris.target[50:]

standardizer = StandardScaler()
#X = standardizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)
X_train_std = standardizer.fit_transform(X_train)
X_test_std = standardizer.fit_transform(X_test)
clfs =[]
#ロジスティック回帰
lg_clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
clfs.append(lg_clf)
# SVM
svc_clf = SVC(gamma='auto')
clfs.append(svc_clf)
#決定木
dt_clf = DecisionTreeClassifier(random_state=0)
clfs.append(dt_clf)

cls_score=[]
cls_ac=[]
cls_prec=[]
cls_rec=[]
cls_f=[]
for clf in clfs:
    clf.fit(X_train_std,y_train)
    pred_y = clf.predict(X_test_std)
    cls_score.append(clf.score(X_test_std,y_test))
    cls_ac.append(accuracy_score(y_test, pred_y))
    cls_prec.append(precision_score(y_test, pred_y))
    cls_rec.append(recall_score(y_test, pred_y))
    cls_f.append(f1_score(y_test, pred_y))


data = {"正答率":cls_ac,"実際のスコア":cls_score,"精度":cls_prec,"検出率":cls_rec,"f値":cls_f}
index = ["ロジスティック回帰","SVM","決定木"]
df = pd.DataFrame(data,index=index)
print(df)

import numpy as np

np.random.seed(seed=0)
n_samples = 500
f0 = [-1, 2]
f1 = [2, -1]
cov = [[1.0,0.8], [0.8, 1.0]]

f0 = np.random.multivariate_normal(f0, cov, int(n_samples/2))
f1 = np.random.multivariate_normal(f1, cov, int(n_samples/2))

X = np.concatenate((f0, f1))
y = np.concatenate((np.ones((int(n_samples/2))), np.ones((int(n_samples/2))) *(-1))).astype(np.int)

random_index = np.random.permutation(np.arange(n_samples))
X = X[random_index]
y = y[random_index]


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
standardizer = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)
X_train_std = standardizer.fit_transform(X_train)
X_test_std = standardizer.fit_transform(X_test)
clfs =[]
#ロジスティック回帰
lg_clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
clfs.append(lg_clf)
# SVM
svc_clf = SVC(gamma='auto')
clfs.append(svc_clf)
#決定木
dt_clf = DecisionTreeClassifier(random_state=0)
clfs.append(dt_clf)

cls_score=[]
cls_ac=[]
cls_prec=[]
cls_rec=[]
cls_f=[]
for clf in clfs:
    clf.fit(X_train_std,y_train)
    pred_y = clf.predict(X_test_std)
    cls_score.append(clf.score(X_test_std,y_test))
    cls_ac.append(accuracy_score(y_test, pred_y))
    cls_prec.append(precision_score(y_test, pred_y))
    cls_rec.append(recall_score(y_test, pred_y))
    cls_f.append(f1_score(y_test, pred_y))
    decision_region(X_train_std,y_train,clf)

data = {"正答率":cls_ac,"実際のスコア":cls_score,"精度":cls_prec,"検出率":cls_rec,"f値":cls_f}
index = ["ロジスティック回帰","SVM","決定木"]
df = pd.DataFrame(data,index=index)
print(df)


X = np.array([[-0.44699 , -2.8073  ],[-1.4621  , -2.4586  ],
       [ 0.10645 ,  1.9242  ],[-3.5944  , -4.0112  ],
       [-0.9888  ,  4.5718  ],[-3.1625  , -3.9606  ],
       [ 0.56421 ,  0.72888 ],[-0.60216 ,  8.4636  ],
       [-0.61251 , -0.75345 ],[-0.73535 , -2.2718  ],
       [-0.80647 , -2.2135  ],[ 0.86291 ,  2.3946  ],
       [-3.1108  ,  0.15394 ],[-2.9362  ,  2.5462  ],
       [-0.57242 , -2.9915  ],[ 1.4771  ,  3.4896  ],
       [ 0.58619 ,  0.37158 ],[ 0.6017  ,  4.3439  ],
       [-2.1086  ,  8.3428  ],[-4.1013  , -4.353   ],
       [-1.9948  , -1.3927  ],[ 0.35084 , -0.031994],
       [ 0.96765 ,  7.8929  ],[-1.281   , 15.6824  ],
       [ 0.96765 , 10.083   ],[ 1.3763  ,  1.3347  ],
       [-2.234   , -2.5323  ],[-2.9452  , -1.8219  ],
       [ 0.14654 , -0.28733 ],[ 0.5461  ,  5.8245  ],
       [-0.65259 ,  9.3444  ],[ 0.59912 ,  5.3524  ],
       [ 0.50214 , -0.31818 ],[-3.0603  , -3.6461  ],
       [-6.6797  ,  0.67661 ],[-2.353   , -0.72261 ],
       [ 1.1319  ,  2.4023  ],[-0.12243 ,  9.0162  ],
       [-2.5677  , 13.1779  ],[ 0.057313,  5.4681  ]])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
standardizer = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)
X_train_std = standardizer.fit_transform(X_train)
X_test_std = standardizer.fit_transform(X_test)
clfs =[]
#ロジスティック回帰
lg_clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
clfs.append(lg_clf)
# SVM
svc_clf = SVC(gamma='auto')
clfs.append(svc_clf)
#決定木
dt_clf = DecisionTreeClassifier(random_state=0)
clfs.append(dt_clf)


cls_score=[]
cls_ac=[]
cls_prec=[]
cls_rec=[]
cls_f=[]
for clf in clfs:
    clf.fit(X_train_std,y_train)
    pred_y = clf.predict(X_test_std)
    cls_score.append(clf.score(X_test_std,y_test))
    cls_ac.append(accuracy_score(y_test, pred_y))
    cls_prec.append(precision_score(y_test, pred_y))
    cls_rec.append(recall_score(y_test, pred_y))
    cls_f.append(f1_score(y_test, pred_y))
    decision_region(X_train_std,y_train,clf)

data = {"正答率":cls_ac,"実際のスコア":cls_score,"精度":cls_prec,"検出率":cls_rec,"f値":cls_f}
index = ["ロジスティック回帰","SVM","決定木"]
df = pd.DataFrame(data,index=index)
print(df)