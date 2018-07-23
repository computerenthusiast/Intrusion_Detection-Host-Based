import numpy as np
import pandas as pd
from statistics import mean

from sklearn.decomposition import PCA

from sklearn import preprocessing
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report,confusion_matrix
path1="C:/Users/rumu/Desktop/new1normallongrowdataset_12hrs.csv"
path2="C:/Users/rumu/Desktop/normal40abnormal60long.csv"
dataset1=pd.read_csv(path1, index_col=0, skipinitialspace=True)
dataset2=pd.read_csv(path2, index_col=0, skipinitialspace=True)
dataset1= dataset1.apply(pd.to_numeric, args=('coerce',))
dataset2= dataset2.apply(pd.to_numeric, args=('coerce',))
dataset1=dataset1.fillna(dataset1.mean())
dataset2=dataset2.fillna(dataset2.mean())

X = dataset1.values[:,:40]
y = dataset2.values[:,21]

kf = KFold(n_splits=10)
kf.get_n_splits(X)
print(kf)
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train_transformed = le.fit_transform(y_train)

print ("Extracting...")
to=time.time()
pca = PCA(n_components=10)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print 'Train pca shape'
print X_train_pca.shape
print 'pca explained_variance_ratio'
print(pca.explained_variance_ratio_)
lda = LinearDiscriminantAnalysis()
lda = lda.fit(X_train_pca, y_train_transformed)
X_train_lda = lda.transform(X_train_pca)
X_test_lda = lda.transform(X_test_pca)
clf = RandomForestClassifier(warm_start=True, oob_score=True,n_estimators=60, verbose=3, random_state=1, max_depth=None, min_samples_leaf=1)
#clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
#clf = LinearDiscriminantAnalysis(solver="svd", shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)
pipe = Pipeline([('pca', pca), ('lda', lda),('RF',clf)])
pipe.fit(X_train_lda,y_train_transformed)
predictions = pipe.predict(X_test_lda)
le1 = preprocessing.LabelEncoder()
le1.fit(y_test)
y_test_transformed = le.fit_transform(y_test)
predictions1 = pipe.predict(X_train_lda)
print accuracy_score(y_train_transformed, predictions1)
print accuracy_score(y_test_transformed, predictions)
print confusion_matrix(y_test_transformed, predictions)
from sklearn.metrics import roc_curve, roc_auc_score 
cl_model = pipe.fit(X_train_lda, y_train_transformed)
y_test_predict_cl = cl_model.predict_proba(X_test_lda)
y_test_scores_lr = [x[1] for x in y_test_predict_cl]
fpr, tpr, thresholds = roc_curve(y_test_transformed, y_test_scores_lr, pos_label=2)
fpr[len(fpr)/2],tpr[len(tpr)/2], thresholds[len(thresholds)/2]
print(classification_report(y_test_transformed,pipe.predict(X_test_lda)))
fpr
thresholds
import matplotlib.pyplot as plt
plt.figure()
plt.title("Feature importances")
sorted_features = sorted(zip(map(lambda x: round(x, 5), clf.feature_importances_), dataset1.columns), reverse=True)
a= pd.DataFrame(sorted_features).to_csv('C:/Users/rumu/Desktop/feature_new_abnormal60normal40long_12hrs_pca_lda_rf.csv',sep=',')
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(X_train_lda.shape[1]):
	print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
plt.bar(range(X_train_lda.shape[1]), importances[indices],color="r", align="center")
plt.xticks(range(X_train_lda.shape[1]), indices)
plt.xlim([-1, X_train_lda.shape[1]])
plt.show()
from collections import OrderedDict
RANDOM_STATE = 1
ensemble_clfs = [
    ("RandomForestClassifier,<br>max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier,<br>max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier,<br>max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)


error_rate
ensemble_clfs = [ ("RandomForestClassifier, max_features=None",RandomForestClassifier(warm_start=True, max_features=None, oob_score=True,))]
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
min_estimators = 10
max_estimators = 100
min_estimators = 10
for label, clf in ensemble_clfs:
	for i in range(min_estimators, max_estimators + 1):
		clf.set_params(n_estimators=i)
		clf.fit(X_train_lda, y_train_transformed)
		oob_error = 1 - clf.oob_score_
		error_rate[label].append((i, oob_error))
		
for label, clf_err in error_rate.items():
	xs, ys = zip(*clf_err)
	plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("no of trees")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()


