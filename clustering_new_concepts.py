import gensim
import tensorflow as tf
import matplotlib
from my_utils import *
# needed imports
from matplotlib import pyplot as plt

import numpy as np
import tokenization
import csv
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

vocab_file = "MODEL/small/vocab.txt"
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

directory_path = "SNO/"
iterations = 100
hier_name = "procedure"
taxonomy = "hier"
vector_model_path = directory_path + "vectorModel/" + hier_name + "/" + str(iterations) + "/"
data_path = directory_path + "data/" + hier_name + "/"

#  PV-DBOW
vector_model_file_0 = vector_model_path + "concept_model0"
pvdbow_model = gensim.models.Doc2Vec.load(vector_model_file_0)

# PV-DM seems better??
vector_model_file_1 = vector_model_path + "concept_model1"
pvdm_model = gensim.models.Doc2Vec.load(vector_model_file_1)

label_file_2017 = directory_path + "data/ontClassLabels_july2017.txt"
conceptLabelDict_2017, _ = read_label(label_file_2017)
label_file_2018 = directory_path + "data/ontClassLabels_jan2018.txt"
conceptLabelDict_2018, _ = read_label(label_file_2018)

import json

jsonFile = data_path + "2018" + hier_name + "_newconcepts_2.json"

test_data = json.load(open(jsonFile))

def get_vector_array(test_data, model):
    concept_vector_arr = []
    concept_id_arr = []
    counter = 0
    for concept_id, value in test_data.items():
        counter += 1
        concept_name = conceptLabelDict_2018[concept_id]
        concept_id_arr.append(concept_id)
        concept_vector = model.infer_vector(tokenizer.tokenize(concept_name))
        concept_vector_arr.append(concept_vector)

    print("total {} new concepts.".format(counter))
    return concept_vector_arr, concept_id_arr

concept_vector_arr, concept_id_arr = get_vector_array(test_data, pvdm_model)

# print(concept_vector_arr)
from scipy.cluster.hierarchy import dendrogram, linkage
# generate the linkage matrix
X = concept_vector_arr
Z = linkage(X, 'ward')
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(concept_vector_arr))
print(c)

print(Z[0])

# calculate full dendrogram
# plt.figure()
plt.figure(figsize=(50, 20))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    labels=concept_id_arr,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=4.,  # font size for the x axis labels
)
plt.tight_layout()
plt.savefig('Hierarchical Clustering Dendrogram.png', bbox_inches='tight')
plt.show()


plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=20,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.tight_layout()
plt.savefig('Hierarchical Clustering Dendrogram (truncated).png', bbox_inches='tight')
plt.show()


# Retrieve the Clusters
# Now, let's finally have a look at how to retrieve the clusters, for different ways of determining k. We can use the fcluster function.
#
# Knowing max_d:
# Let's say we determined the max distance with help of a dendrogram, then we can do the following to get the cluster id for each of our samples:


from scipy.cluster.hierarchy import fcluster
max_d = 15
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)

# Knowing k:
# Another way starting from the dendrogram is to say "i can see i have k=2" clusters. You can then use:


k = 20
clusters = fcluster(Z, k, criterion='maxclust')
print(clusters)

# Using the Inconsistency Method (default):
# If you're really sure you want to use the inconsistency method to determine the number of clusters in your dataset,
#  you can use the default criterion of fcluster() and hope you picked the correct values:

# from scipy.cluster.hierarchy import fcluster
# clusters = fcluster(Z, 8, depth=10)
# print(clusters)

# plt.figure(figsize=(10, 8))
# plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
# plt.show()
from sklearn.manifold import TSNE
import time
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

Y = tsne_results
plt.figure(figsize=(10, 8))
# plt.scatter(Y[:, 0], Y[:, 1], c='red', cmap=plt.cm.Spectral)
plt.scatter(Y[:, 0], Y[:, 1], cmap='rainbow')
plt.show()


