import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
import matplotlib.pyplot as plt

# upload heart disease dataseta
data = pd.read_csv('./heart.csv', sep=',', header=0)

# korelacijska matrica dataseta
corr_matrix = data.corr().abs()
print(corr_matrix)

# razdvajanje atributa x i target-a y
x = data.iloc[:, :-1].values
y = data.iloc[:, 13].values

# random podjela dostupnog seta podataka na trening (85%) i test set (15%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

# normalizacija podataka
# uklanjanje srednje vrijednosti i skaliranje na jedinicu varijanse
scale = StandardScaler()
scale.fit(x_train)

# centralizacija i skaliranje trening i test podataka
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

# KNN klasifikator sa k=7 susjeda
KNNclassifier = KNeighborsClassifier(n_neighbors=7)
KNNclassifier.fit(x_train, y_train)

# SVM klasifikator sa linearnim kernelom
SVCclassifier = SVC(kernel='linear')
SVCclassifier.fit(x_train, y_train)

# testiranje modela, predikcija izlaza
y_predKNN = KNNclassifier.predict(x_test)
y_predSVC = SVCclassifier.predict(x_test)

# matrica konfuzije i rezultat klasifikacije KNN
conf_matrix_KNN = confusion_matrix(y_test, y_predKNN)
class_report_KNN = classification_report(y_test, y_predKNN)

print("\n")
print("KNN Prediction \n")
print(conf_matrix_KNN)
print(class_report_KNN)

# matrica konfuzije i rezultat klasifikacije SVM
conf_matrix_SVM = confusion_matrix(y_test, y_predSVC)
class_report_SVM = classification_report(y_test, y_predSVC)

print("\n")
print("SVM Linear Prediction \n")
print(conf_matrix_SVM)
print(class_report_SVM)

# ROC kriva za KNN i SVM
roc_KNN = plot_roc_curve(KNNclassifier, x_test, y_test)
roc_SVM = plot_roc_curve(SVCclassifier, x_test, y_test)
plt.show()
