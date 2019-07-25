import json
from ml_utils import *



directory_path = "SNO/"
hier_name = "clinical_finding"
data_path = directory_path + "data/" + hier_name + "/"

jsonFile = data_path + "2018" + hier_name + "_newconcepts_2.json"

test_data = json.load(open(jsonFile))

parents_dict = processConceptParents(test_data)
uncles_dict = processConceptUncles(test_data)

all_existing_parents_set = set()
for key, parents in parents_dict.items():
    for parent in parents:
        all_existing_parents_set.add(parent)
positive_lists = []
negative_lists = []
for key, parents in parents_dict.items():
    for parent in parents:
        positive_lists.append([key, parent, 1])
    other_non_parents = list(all_existing_parents_set - set(parents))
    for other_non_parent in other_non_parents:
        negative_lists.append([key, other_non_parent, 0])
print("total count of concept pairs with parents is {}".format(len(positive_lists)))
print("total count of concept pairs with uncles is {}".format(len(negative_lists)))
################################################################################################
# remove duplicates
positive_lists = remove_duplicates(positive_lists)
print("remove duplicates in positive test data, the number is: {}".format(len(positive_lists)))
# remove duplicates
negative_lists = remove_duplicates(negative_lists)
print("remove duplicates in negative test data, the number is: {}".format(len(negative_lists)))
# sampling data otherwise the negative testing sample is too huge
positive_lists, negative_lists = sampling_data(positive_lists, negative_lists)
# data_list, label_list, borrowed_list = read_samples_into_data_label_borrowed(positive_lists, negative_lists)
#
pair_list = positive_lists + negative_lists
shuffle(pair_list)
data_list = []
label_list = []
for line in pair_list:
    data_list.append([line[0], line[1]])
    label_list.append(line[2])