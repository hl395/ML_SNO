import re
import smart_open
import random
import gensim
from pprint import pprint
import numpy as np
from sklearn.utils import shuffle
import json


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return m.group() if m else None


def read_label(fname):
    conceptLabelDict = {}
    errors = []
    with smart_open.smart_open(fname) as f:
        for i, line in enumerate(f):
            # get the id for each concept paragraph
            splitted = line.decode("iso-8859-1").split("\t")
            if len(splitted) == 3:
                conceptID = get_trailing_number(splitted[1])
                conceptLabelDict[conceptID] = splitted[2].replace("\r\n", "")
            else:
                errors.append(splitted)
    return conceptLabelDict, errors


def read_pair(fname):
    # conceptPairDict = {}
    errors = []
    conceptPairList = []
    with smart_open.smart_open(fname) as f:
        for i, line in enumerate(f):
            # get the id for each concept paragraph
            splitted = line.decode("iso-8859-1").split("\t")
            if len(splitted) == 2:
                childID = get_trailing_number(splitted[0])
                parentID = get_trailing_number(splitted[1].replace("\r\n", ""))
                conceptPairList.append([childID, parentID, 1])
            #   conceptPairDict[splitted[1]] = splitted[2].replace("\r\n", "")
            else:
                errors.append(splitted)
    return conceptPairList, errors


def read_not_pair(fname):
    # conceptNotPairDict = {}
    conceptNotPairList = []
    errors = []
    with smart_open.smart_open(fname) as f:
        for i, line in enumerate(f):
            # get the id for each concept paragraph
            splitted = line.decode("iso-8859-1").split("\t")
            if len(splitted) == 2:
                childID = get_trailing_number(splitted[0])
                notparentID = get_trailing_number(splitted[1].replace("\r\n", ""))
                conceptNotPairList.append([childID, notparentID, 0])
            #                 conceptNotPairDict[splitted[1]] = splitted[2].replace("\r\n", "")
            else:
                errors.append(splitted)
    return conceptNotPairList, errors


def remove_duplicates(in_list):
    clean_tuple_list = []
    for pair in in_list:
        t = tuple(pair)
        clean_tuple_list.append(t)
    clean_tuple_list = set(clean_tuple_list)

    clean_pair_list = []
    for t in clean_tuple_list:
        clean_pair_list.append(list(t))
    return clean_pair_list


def leave_for_testing(pair_list_in, not_pair_list_in, number):
    shuffle(pair_list_in)
    shuffle(not_pair_list_in)
    left_pair_list = pair_list_in[: int(number / 2)]
    left_not_pair_list = not_pair_list_in[:int(number / 2)]
    pair_list_in = pair_list_in[int(number / 2):]
    not_pair_list_in = not_pair_list_in[int(number / 2):]
    testing_pair_lists = [*left_pair_list, *left_not_pair_list]
    return pair_list_in, not_pair_list_in, testing_pair_lists


def sampling_data(concept_pair_list, concept_not_pair_list, times=1):
    if len(concept_pair_list) < len(concept_not_pair_list):
        print("Downsampling negative samples")
        # Downsampling negative samples
        shuffle(concept_not_pair_list)
        # leftPairList = concept_not_pair_list[len(concept_pair_list):]
        concept_not_pair_list = concept_not_pair_list[:len(concept_pair_list)*times]
    else:
        print("Upsampling negative samples")
        # Upsampling negative samples
        while len(concept_pair_list) != len(concept_not_pair_list):
            shuffle(concept_not_pair_list)
            duplicatedList = concept_not_pair_list[:len(concept_pair_list) - len(concept_not_pair_list)]
            print(len(duplicatedList))
            shuffle(concept_not_pair_list)
            concept_not_pair_list.extend(duplicatedList)

    print("Number of pairs: ", len(concept_pair_list))
    print("Number of not pairs: ", len(concept_not_pair_list))
    assert len(concept_pair_list)*times == len(concept_not_pair_list), "Mismatch in Positive & Negative samples!"
    return concept_pair_list, concept_not_pair_list


def load_vector_model(vector_model_file):
    vector_model = gensim.models.Doc2Vec.load(vector_model_file)
    inferred_vector_0 = vector_model.infer_vector(
        ['congenital', 'prolong', 'rupture', 'premature', 'membrane', 'lung'])
    pprint(vector_model.docvecs.most_similar([inferred_vector_0], topn=10))
    return vector_model


def readFromPairList(id_pair_list, id_notPair_list):
    pair_list = id_pair_list + id_notPair_list
    shuffle(pair_list)
    idpairs_list = []
    label_list = []
    for i, line in enumerate(pair_list):
        idpairs_list.append([line[0], line[1]])
        label_list.append(line[2])
    return idpairs_list, label_list


def getVectorFromModel(concept_id, conceptLabelDict, model, opt_str=""):
    if concept_id in model.docvecs:
        concept_vector = model.docvecs[concept_id]
    else:
        # print("%s not found, get inferred vector "%(concept_id))
        concept_label = conceptLabelDict[concept_id]
        line = opt_str + concept_label
        tokens = gensim.utils.simple_preprocess(line)
        concept_vector = model.infer_vector(tokens)
        # concept_vector = model.infer_vector(opt_str.split() + concept_label.split())
    return concept_vector


def getVector(line, conceptLabelDict, model, opt_str=""):
    a = getVectorFromModel(line[0], conceptLabelDict, model, opt_str)
    b = getVectorFromModel(line[1], conceptLabelDict, model, opt_str)
    c = np.array((a, b))
    c = c.T
    #     c = np.expand_dims(c, axis=2)
    #     print(c.shape)
    return c


# stack vectors into 4 channels

def stack_vector(pvdm_vector, pvdbow_vector, one_vector_flag):
    if one_vector_flag:
        return pvdm_vector
    else:
        return np.concatenate((pvdm_vector, pvdbow_vector), axis=1)

def stack_vector_for_multiple_vectors(vector_list, one_vector_flag):
    if one_vector_flag:
        return vector_list[0]
    else:
        return np.concatenate(np.asarray(vector_list), axis=1)


def get_batches(x_samples, y_samples, conceptLabelDict, pvdm_model, pvdbow_model, one_vector_flag, batch_size=64, n_classes=2, random_flag=True,  op_str=''):
    samples = list(zip(x_samples, y_samples))
    num_samples = len(samples)
    if random_flag:
        shuffle(samples)
    for offset in range(0, num_samples, batch_size):
        batch_samples = samples[offset:offset + batch_size]

        X_samples = []
        Y_samples = []
        for batch_sample in batch_samples:
            pair_list = batch_sample[0]
            # data_vector = getVector(pair_list, conceptLabelDict, vector_model)
            pvdm_vector = getVector(pair_list, conceptLabelDict, pvdm_model, op_str)
            pvdbow_vector = getVector(pair_list, conceptLabelDict, pvdbow_model, op_str)
            data_vector = stack_vector(pvdm_vector=pvdm_vector, pvdbow_vector=pvdbow_vector, one_vector_flag=one_vector_flag)
            # data_vector = stackVector(data_vector)
            # print(data_vector.shape)
            X_samples.append(data_vector)
            class_label = batch_sample[1]
            Y_samples.append(class_label)

        X_samples = np.array(X_samples).astype('float32')
        Y_samples = np.eye(n_classes)[Y_samples]
        #             print('one batch ready')
        if random_flag:
            yield shuffle(X_samples, Y_samples)
        else:
            yield (X_samples, Y_samples)

def get_batches_for_mulitple_vectors(x_samples, y_samples, conceptLabelDict, vector_models_list, one_vector_flag, batch_size=64, n_classes=2, random_flag=True,  op_str=''):
    samples = list(zip(x_samples, y_samples))
    num_samples = len(samples)
    if random_flag:
        shuffle(samples)
    for offset in range(0, num_samples, batch_size):
        batch_samples = samples[offset:offset + batch_size]

        X_samples = []
        Y_samples = []
        for batch_sample in batch_samples:
            pair_list = batch_sample[0]
            vector_list = []
            for vector_model in vector_models_list:
                # data_vector = getVector(pair_list, conceptLabelDict, vector_model)
                vector_list.append(getVector(pair_list, conceptLabelDict, vector_model, op_str))
            data_vector = stack_vector_for_multiple_vectors(vector_list, one_vector_flag=one_vector_flag)
            # data_vector = stackVector(data_vector)
            # print(data_vector.shape)
            X_samples.append(data_vector)
            class_label = batch_sample[1]
            Y_samples.append(class_label)

        X_samples = np.array(X_samples).astype('float32')
        Y_samples = np.eye(n_classes)[Y_samples]
        #             print('one batch ready')
        if random_flag:
            yield shuffle(X_samples, Y_samples)
        else:
            yield (X_samples, Y_samples)


def readFromJsonData2(test_data, conceptLabelDict):
    result_paired = []
    result_not_paired = []
    for key, value in test_data.items():
        if value['Parents']:
            for x in range(len(value['Parents'])):
                if value['Parents'][x] in conceptLabelDict:
                    result_paired.append([key, value['Parents'][x], 1])
        if value['Siblings']:
            for x in range(len(value['Siblings'])):
                if value['Siblings'][x] in conceptLabelDict:
                    result_not_paired.append([key, value['Siblings'][x], 0])
        if value['Children']:
            for x in range(len(value['Children'])):
                if value['Children'][x] in conceptLabelDict:
                    result_not_paired.append([key, value['Children'][x], 0])
        return result_paired, result_not_paired


def readFromJsonData(test_data):
    result_paired = []
    result_not_paired = []
    for key, value in test_data.items():
        if value['Parents']:
            for x in range(len(value['Parents'])):
                result_paired.append([key, value['Parents'][x], 1])
        if value['Siblings']:
            for x in range(len(value['Siblings'])):
                result_not_paired.append([key, value['Siblings'][x], 0])
        # if value['Children']:
        #  for x in range(len(value['Children'])):
        #    result_not_paired.append([key, value['Children'][x], 0])
    return result_paired, result_not_paired


def processConceptParents(json_data):
    parents_dict = {}
    for key, value in json_data.items():
        parents_list = []
        if value['Parents']:
            for x in range(len(value['Parents'])):
                parents_list.append(value['Parents'][x])
        parents_dict[key] = parents_list
    return parents_dict


def processConceptChildren(json_data):
    children_dict = {}
    for key, value in json_data.items():
        children_list = []
        if value['Children']:
            for x in range(len(value['Children'])):
                children_list.append(value['Children'][x])
        children_dict[key] = children_list
    return children_dict


def readOutPairsTwo(key, parent, paired_list, not_paired_list):
    pair_list = []
    not_pair_list = []
    for i in range(len(paired_list)):
        pair = paired_list[i]
        if pair[0] == key and pair[1] != parent:
            pair_list.append(pair)
    for i in range(len(not_paired_list)):
        pair = not_paired_list[i]
        if pair[0] == key:
            not_pair_list.append(pair)
    return pair_list, not_pair_list


def from_testing_remove_training_samples(pair_list, not_pair_list, testing_list):
    # from testing samples remove samples that are already been used for training samples
    # convert pair list to pair set
    print("convert pair list to pair set")
    train_pair_list = []
    for pair in pair_list:
        pair = tuple(pair)
        train_pair_list.append(pair)
    train_pair_set = set(train_pair_list)

    # convert not_pair list to not_pair set
    print("convert not_pair list to not_pair set")
    train_not_pair_list = []
    for pair in not_pair_list:
        pair = tuple(pair)
        train_not_pair_list.append(pair)
    train_not_pair_set = set(train_not_pair_list)

    # convert testing list to testing set
    print("convert testing list to testing set")
    testing_pair_list = []
    for pair in testing_list:
        pair = tuple(pair)
        testing_pair_list.append(pair)
    testing_pair_set = set(testing_pair_list)

    # remove testing samples already in training
    print("remove testing samples already in training")
    rest_pair_set = testing_pair_set - train_pair_set - train_not_pair_set
    # type(rest_pair_Set)

    # convert testing set back to testing list
    print("convert testing set back to testing list")
    rest_pair_list = list(rest_pair_set)
    rest_testing_pair_list = []
    for t in rest_pair_list:
        rest_testing_pair_list.append(list(t))
    return rest_testing_pair_list


def read_leaveout_from_file(leaveout_file):
    # read the randomly chosen 2000 positive and 2000 negative samples for testing
    with open(leaveout_file) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    testing_pair_lists = []
    for line in content:
        item_list = line.split('\t')
        item_list[2] = int(item_list[2])
        testing_pair_lists.append(item_list)
    return testing_pair_lists


def read_sample_from_string(test_string):
    sample = []
    label = []
    for line in test_string.splitlines():
        word = line.strip().replace("\r\n\t", "").split(",")
        sample.append([word[0], word[1]])
        label.append(int(word[2]))
    return sample, label


def remove_testing_samples_from_training(testing_list, sample_list, label_list):
    for sample, label in zip(sample_list, label_list):
        if sample in testing_list:
            print("Sample {} will be used for testing, thus remove from training".format(sample))
            sample_list.remove(sample)
            label_list.remove(label)
    print("after remove testing sample from training {} ".format(len(sample_list)))
    print("after remove testing label from training {} ".format(len(label_list)))
    return sample_list, label_list


def read_test_pair(fname, model):
    testingPairList = []
    errors = []
    with smart_open.smart_open(fname) as f:
        for i, line in enumerate(f):
            # get the id for each concept paragraph
            splitted = line.decode("iso-8859-1").split("\t")
            if len(splitted)==2:
                childID = get_trailing_number(splitted[0])
                notparentID = get_trailing_number(splitted[1].replace("\r\n", ""))
                assert childID in model.docvecs, "%s not in vector model"%(childID)
                assert notparentID in model.docvecs, "%s not in vector model"%(notparentID)
                testingPairList.append([childID, notparentID, 0])
            # conceptNotPairDict[splitted[1]] = splitted[2].replace("\r\n", "")
            else:
                errors.append(splitted)
    return testingPairList, errors

# test one pair at a time, make sure the order is right
def get_one_vector_batch(pair, label, conceptLabelDict, pvdm_model, pvdbow_model, one_vector_flag, n_classes=2):
    print("{} : {} ".format(pair, label))
    print("{} -> {} ".format(conceptLabelDict[pair[0]], conceptLabelDict[pair[1]]))
#            data_vector= getVector(pair_list, conceptLabelDict, vector_model)
    pvdm_vector = getVector(pair, conceptLabelDict, pvdm_model)
    pvdbow_vector = getVector(pair, conceptLabelDict, pvdbow_model)
    data_vector = stack_vector(pvdm_vector=pvdm_vector, pvdbow_vector=pvdbow_vector, one_vector_flag=one_vector_flag)
    data_vector = np.array(data_vector).astype('float32')
    class_label = np.eye(n_classes)[label]
    data_vector = np.expand_dims(data_vector, axis=0)
    class_label = np.expand_dims(class_label, axis=0)
    return data_vector, class_label


def plot_confusion_matrix(cls_true, cls_pred, img_path, img_name, num_classes=2):
    from sklearn.metrics import confusion_matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(img_path + img_name +'confusion_matrix.png')
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def read18FromJsonData(test_data):
    result_paired = []
    result_not_paired = []
    for key, value in test_data.items():
        print("\nConcepts: ", key)
        if value['Parea Uncles']:
            for parea_uncle in value['Parea Uncles']:
                # print("parea_uncle {} : {}".format(key, parea_uncle))
                result_not_paired.append([key, parea_uncle, 0])
        if value['Area Uncles']:
            for area_uncle in value['Area Uncles']:
                # print("area_uncle {} : {}".format(key, area_uncle))
                result_not_paired.append([key, area_uncle, 0])
        if value['Uncles']:
            for uncle in value['Uncles']:
                # print("uncle {} : {}".format(key, uncle))
                result_not_paired.append([key, uncle, 0])
        if value['Parents']:
            for parent in value['Parents']:
                # print("parent {} : {}".format(key, parent))
                result_paired.append([key, parent, 1])

    for i, element in enumerate(result_not_paired):
        result_not_paired[i] = [get_trailing_number(element[0]), get_trailing_number(element[1]), 0]

    for i, element in enumerate(result_paired):
        result_paired[i] = [get_trailing_number(element[0]), get_trailing_number(element[1]), 1]

    return result_paired, result_not_paired


def read18SiblingsFromJsonData(test_data):
    result_not_paired = []
    for key, value in test_data.items():
        print("\nConcepts: ", key)
        if value['Siblings']:
            for sibling in value['Siblings']:
                # print("sibling {} : {}".format(key, sibling))
                result_not_paired.append([key, sibling, 0])
    for i, element in enumerate(result_not_paired):
        result_not_paired[i] = [get_trailing_number(element[0]), get_trailing_number(element[1]), 0]
    return result_not_paired


def remove_conflicts(data_list, label_list, testing_data_list):
    for data, label in zip(data_list, label_list):
        if data in testing_data_list:
            data_list.remove(data)
            label_list.remove(label)
    return data_list, label_list


def prepare_negative_testing_data(n_data_list, n_label_list, sibling_data_list, sibling_label_list, testing_data_list):
    negative_data_list = []
    negative_label_list =[]
    for x in testing_data_list:
        internal_result=[]
        for i, l in zip(n_data_list, n_label_list):
            if x[0] == i[0] and l == 0:
                internal_result.append([i,l])
        if internal_result:
            internal_result = shuffle(internal_result)
            # print(internal_result[0])
            negative_data_list.append(internal_result[0][0])
            negative_label_list.append(internal_result[0][1])
            n_data_list.remove(internal_result[0][0])
            n_label_list.remove(internal_result[0][1])
        else:
            sibling_result = []
            for i, l in zip(sibling_data_list, sibling_label_list):
                if x[0] == i[0] and l == 0:
                    sibling_result.append([i, l])
            if sibling_result:
                sibling_result = shuffle(sibling_result)
                negative_data_list.append(sibling_result[0][0])
                negative_label_list.append(sibling_result[0][1])
    return negative_data_list, negative_label_list, n_data_list, n_label_list


def append_list(n_data_list, n_label_list, data_list, label_list ):
    for d, l in zip(n_data_list, n_label_list):
        if d not in data_list:
            data_list.append(d)
            label_list.append(l)
    return data_list, label_list


################################################################################################
# Methods section for SNO only
def sno_get_batches_for_testing_new(x_samples, y_samples, conceptLabelDict, pvdm_model, pvdbow_model, one_vector_flag,
                                    batch_size=64, n_classes=2, random_flag=True,  op_str='', borrowed_list=[]):
    if borrowed_list:
        samples = list(zip(x_samples, y_samples, borrowed_list))
    else:
        samples = list(zip(x_samples, y_samples))
    num_samples = len(samples)
    if random_flag:
        shuffle(samples)
    for offset in range(0, num_samples, batch_size):
        batch_samples = samples[offset:offset + batch_size]

        X_samples = []
        Y_samples = []
        for batch_sample in batch_samples:
            pair_list = batch_sample[0]
            if borrowed_list:
                op_str = conceptLabelDict[batch_sample[2]]
            # data_vector = getVector(pair_list, conceptLabelDict, vector_model)
            pvdm_vector = getVector(pair_list, conceptLabelDict, pvdm_model, op_str)
            pvdbow_vector = getVector(pair_list, conceptLabelDict, pvdbow_model, op_str)
            data_vector = stack_vector(pvdm_vector=pvdm_vector, pvdbow_vector=pvdbow_vector, one_vector_flag=one_vector_flag)
            # data_vector = stackVector(data_vector)
            # print(data_vector.shape)
            X_samples.append(data_vector)
            class_label = batch_sample[1]
            Y_samples.append(class_label)

        X_samples = np.array(X_samples).astype('float32')
        Y_samples = np.eye(n_classes)[Y_samples]
        #             print('one batch ready')
        if random_flag:
            yield shuffle(X_samples, Y_samples)
        else:
            yield (X_samples, Y_samples)

def sno_get_batches_for_testing_new_for_multiple_vectors(x_samples, y_samples, conceptLabelDict, vector_models_list, one_vector_flag,
                                    batch_size=64, n_classes=2, random_flag=True,  op_str='', borrowed_list=[]):
    if borrowed_list:
        samples = list(zip(x_samples, y_samples, borrowed_list))
    else:
        samples = list(zip(x_samples, y_samples))
    num_samples = len(samples)
    if random_flag:
        shuffle(samples)
    for offset in range(0, num_samples, batch_size):
        batch_samples = samples[offset:offset + batch_size]

        X_samples = []
        Y_samples = []
        for batch_sample in batch_samples:
            pair_list = batch_sample[0]
            if borrowed_list:
                op_str = conceptLabelDict[batch_sample[2]]
            vector_list = []
            for vector_model in vector_models_list:
                vector_list.append(getVector(pair_list, conceptLabelDict, vector_model, op_str))
            data_vector = stack_vector_for_multiple_vectors(vector_list, one_vector_flag=one_vector_flag)
            X_samples.append(data_vector)
            class_label = batch_sample[1]
            Y_samples.append(class_label)

        X_samples = np.array(X_samples).astype('float32')
        Y_samples = np.eye(n_classes)[Y_samples]
        #             print('one batch ready')
        if random_flag:
            yield shuffle(X_samples, Y_samples)
        else:
            yield (X_samples, Y_samples)


def find_existing_parents_set_for_all_new_concepts(parents_dict, conceptLabelDict):
    all_existing_parents_set = set()
    non_existing_parents_set = set()
    for key, parents in parents_dict.items():
        # print("Processing key ", key)
        for parent in parents:
            if parent in conceptLabelDict:
                all_existing_parents_set.add(parent)
            else:
                non_existing_parents_set.add(parent)
    print("all existing parents number is {}".format(len(all_existing_parents_set)))
    return all_existing_parents_set, non_existing_parents_set


def find_concepts_with_existing_parents(parents_dict, conceptLabelDict):
    existing_parents_dict = {}
    counter = 0
    no_existing_pairs_counter = 0
    total_pairs_counter = 0
    parent_not_existing_pairs_counter = 0
    for key, parents in parents_dict.items():
        flag = True
        existing_parents_list = []
        for parent in parents:
            if parent in conceptLabelDict:
                total_pairs_counter += 1
                flag = False
                existing_parents_list.append(parent)
            else:
                parent_not_existing_pairs_counter += 1
        if existing_parents_list:
            existing_parents_dict[key] = existing_parents_list
        if flag:
            counter += 1
            no_existing_pairs_counter += len(parents)
    print("total concepts with existing parents count is {}".format(len(existing_parents_dict)))
    print("number of concepts with no existing parents is {}".format(counter))
    print("total number of child-parent pairs is {}".format(total_pairs_counter))
    parent_not_existing_pairs_counter = parent_not_existing_pairs_counter - no_existing_pairs_counter
    print("total number of not existing parents child-parent pairs for concept have existing parents is {}".format(parent_not_existing_pairs_counter))
    print("total number of not existing parents child-parent pairs for concept have no existing parents is {}".format(no_existing_pairs_counter))
    return existing_parents_dict

def find_concepts_with_existing_uncles(uncles_dict, conceptLabelDict):
    existing_uncles_dict = {}
    for key, uncles in uncles_dict.items():
        existing_uncles_list = []
        for uncle in uncles:
            if uncle in conceptLabelDict:
                existing_uncles_list.append(uncle)
        existing_uncles_dict[key] = existing_uncles_list
    print("total concepts with existing uncles count is {}".format(len(existing_uncles_dict)))
    return existing_uncles_dict


def prepare_samples_for_concepts_with_multiple_existing_parents(existing_parents_dict, all_existing_parents_set):
    positive_lists = []
    negative_lists = []
    for key, parents in existing_parents_dict.items():
        if len(parents) >= 2:
            for parent in parents:
                borrowed_parent = parent
                other_parents = [x for x in parents if x!= parent]
                for other_parent in other_parents:
                    positive_lists.append([key, other_parent, 1, borrowed_parent])

                other_non_parents = list(all_existing_parents_set - set(parents))
                for other_non_parent in other_non_parents:
                    negative_lists.append([key, other_non_parent, 0, borrowed_parent])
    return positive_lists, negative_lists

def prepare_samples_for_concepts_with_multiple_existing_parents_and_uncles(existing_parents_dict, existing_uncles_dict):
    positive_lists = []
    negative_lists = []
    for key, parents in existing_parents_dict.items():
        if len(parents) >= 2:
            for parent in parents:
                borrowed_parent = parent
                other_parents = [x for x in parents if x!= parent]
                for other_parent in other_parents:
                    positive_lists.append([key, other_parent, 1, borrowed_parent])

                if key in existing_uncles_dict:
                    for existing_uncle in existing_uncles_dict[key]:
                        negative_lists.append([key, existing_uncle, 0, borrowed_parent])
    return positive_lists, negative_lists


def read_samples_into_data_label_borrowed(positive_lists, negative_lists):
    pair_list = positive_lists + negative_lists
    shuffle(pair_list)
    data_list = []
    label_list = []
    borrowed_list = []
    for line in pair_list:
        data_list.append([line[0], line[1]])
        label_list.append(line[2])
        borrowed_list.append(line[3])
    return data_list, label_list, borrowed_list


def processConceptUncles(json_data):
    uncles_dict = {}
    for key, value in json_data.items():
        uncles_list = []
        if value['Uncles']:
            for x in range(len(value['Uncles'])):
                uncles_list.append(value['Uncles'][x])
            uncles_dict[key] = uncles_list
    return uncles_dict
