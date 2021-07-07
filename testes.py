from sklearn import datasets

iris = datasets.load_iris()

#comprimento sépala em cm
#largura sépala em cm
#comprimento da pétala em cm
#largura da pétala em cm 

X = iris.data[:, 0:4]
y = iris.target

from sklearn.naive_bayes import GaussianNB
r = GaussianNB()
r.fit(X, y)
result = r.predict([[4, 2.5, 7.4, 3.4]])
print(result)
print(iris.target_names[result])
#setosa, versicular, virginica