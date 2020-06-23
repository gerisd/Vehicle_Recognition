'''
Script finds the model with the best score and trains that model on the results outputed by the ResNet model 
and stores it to pickle file

'''
import pickle
import h5py

from sklearn import preprocessing, svm
from sklearn.metrics import r2_score

#Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


train_input = h5py.File(r"./results/ResNetOutput.hdf5", "r")
val_input = h5py.File(r"./results/ResNetValOutput.hdf5", "r")

car_index = pickle.loads(open(r"./results/class_index.pickle", "rb").read())
car_labels = pickle.loads(open(r"./results/class_names.pickle", "rb").read())

with h5py.File(r"./results/ResNetValOutput.hdf5", "r") as f:
	a_group_key = list(f.keys())[0]

	#get the data
	data = list(f[a_group_key])
	

#make the model - experiement using multiple different ones and experiment which one is better
class_name = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(7),
    svm.SVC(kernel="linear", C=1.0, probability=True),
    svm.SVC(gamma=2, C=1),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

#Classification dict that stores the scores of each classifier
#Select classifier with highest score
#Using a validation test set 
class_score = {}
for cl_name, classifier in zip(class_name, classifiers):
	classifier.fit(train_input["features"], train_input['labels'])
	pred = classifier.predict(val_input["features"])
	score = classifier.score(val_input["features"], val_input['labels'])
	print(f"Name: {cl_name}, Score: {score}")
	class_score[cl_name] = score

#Find models with the highest score
high_score = 0
best_models = {}
for key, value in sorted(class_score.items(), key = lambda kv: kv[1], reverse=True):
	if value >= high_score:
		high_score = value
		best_models[key] = value

#Selecting one of the best performed models and storing the model to disk 
top_model = next(iter(best_models))	

#Get model 
model_index = class_name.index(top_model)
clf = classifiers[model_index] 

clf.fit(train_input['features'], names)

#Store the trained model to a pickle file
with open("model.pickle", "wb") as handle:
	pickle.dump(clf, handle, protocol = pickle.HIGHEST_PROTOCOL)

with open("le.pickle", "wb") as handle:
    pickle.dump(le, handle, protocol = pickle.HIGHEST_PROTOCOL)

#close files
train_input.close()
val_input.close()
