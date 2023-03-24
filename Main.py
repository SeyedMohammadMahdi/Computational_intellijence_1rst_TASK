from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from warnings import filterwarnings
filterwarnings("ignore")

# loading iris dataset
iris = load_iris()
# define a list for different k
# 120 is to test if the verification method cares about number of clusters
# if the number of clusters is equal to dataset then all datas will be is a cluster with only one member
# in this case if we only count the most label in each cluster, the clustering correct members rate will be 100%
# although the clustering is aweful
K = [2, 3, 4, 5, 6, 7, 8, 9, 10, 120]

# separating data and target
data = iris.data
target = iris.target

# spliting data into train(80%) and test(20%)
trainData, testData, trainTarget, testTarget = train_test_split(data, target, test_size=.2, random_state=42)
# randIndex stores the value of rand index for each k
randIndex = []

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(trainData)
    randIndex.append(adjusted_rand_score(trainTarget, kmeans.labels_)) 

print(randIndex)