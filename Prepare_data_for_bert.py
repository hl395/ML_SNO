# coding: utf-8

from my_utils import *



######################################################################################################


def write_train_tsv(filename, records):
    with open(filename, "w", encoding='utf-8') as record_file:
        record_file.write("Quality\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for record in records:
            record_file.write(str(record))

def write_test_tsv(filename, records):
    with open(filename, "w", encoding='utf-8') as record_file:
        record_file.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for record in records:
            record_file.write(str(record))


hier_list = ["procedure", "clinical_finding"]  #
negative_flags = ["hier", "area", "parea"]  # []
c_list = [(x, y) for x in hier_list for y in negative_flags]

# global variables
# hier_name = "clinical_finding"
# hier_name = "procedure"
# directory_path = "/home/h/hl395/mlontology/SNO/"
directory_path = "SNO/"    # debug ****

#############################################################################################################
for hier_name, negative_flag in c_list:

    print("Run training with hierarchy: {} ".format(hier_name))
    print("Testing with negative taxonomy {}".format(negative_flag))

    data_path = directory_path + "data/" + hier_name + "/"

    # negative_flag = "area"  #
    if negative_flag == "hier":

        notPair_file = data_path + "taxNotPairs_sno_" + hier_name + "_hier.txt"

    elif negative_flag == "area":

        notPair_file = data_path + "taxNotPairs_sno_" + hier_name + "_area.txt"

    else:

        notPair_file = data_path + "taxNotPairs_sno_" + hier_name + "_parea.txt"


    positive_flag = "hier"  #
    if positive_flag == "hier":
        pair_file = data_path + "taxPairs_sno_" + hier_name + "_hier.txt"
    elif positive_flag == "area":
        pair_file = data_path + "taxPairs_sno_" + hier_name + "_area.txt"
    else:
        pair_file = data_path + "taxPairs_sno_" + hier_name + "_parea.txt"

    print("Positive training data from {}".format(positive_flag))
    print("Negative training data from {}".format(negative_flag))

    label_file_2017 = directory_path + "data/ontClassLabels_july2017.txt"
    conceptLabelDict_2017, _ = read_label(label_file_2017)
    label_file_2018 = directory_path + "data/ontClassLabels_jan2018.txt"
    conceptLabelDict_2018, _ = read_label(label_file_2018)

    # read positive samples
    conceptPairList, _ = read_pair(pair_file)
    checkpairs = conceptPairList[10:15]
    print(checkpairs)
    print("number of pairs: ", len(conceptPairList))

    # read negative samples
    conceptNotPairList, _ = read_not_pair(notPair_file)
    checkNonpairs = conceptNotPairList[10:15]
    print(checkNonpairs)
    print("number of not pairs: ", len(conceptNotPairList))

    # remove duplicates
    print("remove duplicates")
    conceptPairList = remove_duplicates(conceptPairList)
    print("After remove duplicates in linked pairs: ")
    print(len(conceptPairList))
    conceptNotPairList = remove_duplicates(conceptNotPairList)
    print("After remove duplicates in not linked pairs: ")
    print(len(conceptNotPairList))

    # leave out 2000 positive and 2000 negative samples for testing
    conceptPairList, conceptNotPairList, testing_pair_lists = leave_for_testing(conceptPairList, conceptNotPairList,
                                                                                4000)

    conceptPairList, conceptNotPairList = sampling_data(conceptPairList, conceptNotPairList)



    # read both negative and positive pairs into pairs list and label list
    idpairs_list, label_list = readFromPairList(conceptPairList, conceptNotPairList)
    print(label_list[:20])

    # split samples into training and validation set
    from sklearn.model_selection import train_test_split

    X_train, X_validation, y_train, y_validation = train_test_split(idpairs_list, label_list, test_size=0.1,
                                                                    shuffle=True)
    print(len(X_train))
    print(len(X_validation))
    print(X_train[:20])
    print(X_validation[:20])
    print(y_train[:20])
    print(y_validation[:20])


    lines = []
    for xtr, ytr in zip(X_train, y_train):
        # print("xtr is {}\t ytr is {}".format(xtr, ytr))
        line = "{}\t{}\t{}\t{}\t{}\n".format(ytr, xtr[1], xtr[0], conceptLabelDict_2017[xtr[1]], conceptLabelDict_2017[xtr[0]])
        lines.append(line)


    filename = hier_name + '_' + negative_flag +'_train.tsv'
    print(len(lines))
    # if (os.path.exists(filename) == False):
    #     os.makedirs(filename)
    write_train_tsv(filename, lines)

    # lines = []
    # for xtr, ytr in zip(X_validation, y_validation):
    #     # print("xtr is {}\t ytr is {}".format(xtr, ytr))
    #     line = "{}\t{}\t{}\t{}\t{}\n".format(ytr, xtr[0], xtr[1], conceptLabelDict_2017[xtr[0]], conceptLabelDict_2017[xtr[1]])
    #     lines.append(line)
    #
    # filename = 'dev.tsv'
    # print(len(lines))
    # write_train_tsv(filename, lines)

    print("result for testing leave-out samples: ")
    idpairs_list, label_list = readFromPairList([], testing_pair_lists)

    lines = []
    for xtr, ytr in zip(idpairs_list, label_list):
        # print("xtr is {}\t ytr is {}".format(xtr, ytr))
        line = "{}\t{}\t{}\t{}\t{}\n".format(ytr, xtr[1], xtr[0], conceptLabelDict_2017[xtr[1]],
                                             conceptLabelDict_2017[xtr[0]])
        lines.append(line)

    filename = hier_name + '_' + negative_flag +'_dev.tsv'
    print(len(lines))
    write_train_tsv(filename, lines)




    import json

    jsonFile = data_path + "2018" + hier_name + "_newconcepts_2.json"

    test_data = json.load(open(jsonFile))
    parents_dict = processConceptParents(test_data)
    uncles_dict = processConceptUncles(test_data)
    ##################################################################
    # positive_lists = []
    # for key, parents in parents_dict.items():
    #     for parent in parents:
    #         positive_lists.append([key, parent, 1])
    # negative_lists = []
    # for key, uncles in uncles_dict.items():
    #     for uncle in uncles:
    #         negative_lists.append([key, uncle, 0])
    # print("total count of concept pairs with parents is {}".format(len(positive_lists)))
    # print("total count of concept pairs with uncles is {}".format(len(negative_lists)))
    ###############################################################################################
    # existing_parents_dict = find_concepts_with_existing_parents(parents_dict, conceptLabelDict_2017)
    # existing_uncles_dict = find_concepts_with_existing_uncles(uncles_dict, conceptLabelDict_2017)
    # positive_lists, negative_lists = prepare_samples_for_concepts_with_multiple_existing_parents_and_uncles(
    #     existing_parents_dict, existing_uncles_dict)
    ################################################################################################
    # all_existing_parents_set = set()
    # for key, parents in parents_dict.items():
    #     for parent in parents:
    #         all_existing_parents_set.add(parent)

    all_existing_parents_set = find_concepts_with_existing_parents(parents_dict, conceptLabelDict_2017)
    positive_lists = []
    negative_lists = []
    for key, parents in parents_dict.items():
        for parent in parents:
            if parent in all_existing_parents_set:
                positive_lists.append([key, parent, 1])
        other_non_parents = list(set(all_existing_parents_set) - set(parents))
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



    index = 0
    lines = []
    for xtr, ytr in zip(data_list, label_list):
        # print("xtr is {}\t ytr is {}".format(xtr, ytr))
        line = "{}\t{}\t{}\t{}\t{}\t{}\n".format(index, xtr[1], xtr[0], conceptLabelDict_2018[xtr[1]], conceptLabelDict_2018[xtr[0]], ytr)
        index += 1
        lines.append(line)

    filename = hier_name + '_' + negative_flag +'_test.tsv'
    print(len(lines))
    write_test_tsv(filename, lines)
    # break



print("testing done")