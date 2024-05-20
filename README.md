import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
import pydot
df = pd.read_csv('/content/kyphosis (1).csv')
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']
X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.20)
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print("Accuracy Score", accuracy_score(y_test, predictions))
print("Confusion Matrix")
print(confusion_matrix(y_test,predictions))
#Visualization
features = list(df.columns[1:])
dot_data = StringIO()
export_graphviz(dtree,out_file=dot_data,feature_names=features,
filled=True,rounded=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())
