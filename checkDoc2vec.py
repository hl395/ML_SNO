import gensim
from my_utils import *

directory_path = "SNO/"
iterations = 100
hier_name = "procedure"
# hier_name = "clinical_finding"
taxonomy = "hier"
vector_model_path = directory_path + "vectorModel/" + hier_name + "/" + str(iterations) + "/"
data_path = directory_path + "data/" + hier_name + "/"

#  PV-DBOW
vector_model_file_0 = vector_model_path + hier_name + "_512_concept_model0"
pvdbow_model = gensim.models.Doc2Vec.load(vector_model_file_0)

# PV-DM seems better??
vector_model_file_1 = vector_model_path + hier_name + "_512_concept_model1"
pvdm_model = gensim.models.Doc2Vec.load(vector_model_file_1)


label_file_2017 = directory_path + "data/ontClassLabels_july2017.txt"
conceptLabelDict_2017, _ = read_label(label_file_2017)
label_file_2018 = directory_path + "data/ontClassLabels_jan2018.txt"
conceptLabelDict_2018, _ = read_label(label_file_2018)

print(conceptLabelDict_2017['446965002'])

concept_id = '71388002'

if concept_id in pvdm_model.docvecs:
    concept_vector = pvdm_model.docvecs[concept_id]
    print(concept_vector)

concept_id = '91130003'

if concept_id in pvdbow_model.docvecs:
    concept_vector = pvdm_model.docvecs[concept_id]
    print(concept_vector)