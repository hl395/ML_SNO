# coding: utf-8
import os
import time
import tensorflow as tf
import matplotlib
from ml_utils import *

# global setting
vector_type = "2vectors"  # 1vector
print("Processing {}".format(vector_type))
if vector_type == "1vector":
    one_vector_flag = True
else:
    one_vector_flag = False

######################################################################################################
# set model parameters
batch_size = 4000  # Batch size : 4000  debug ****
seq_len = 512  # word embedding length
learning_rate = 0.0001
lambda_loss_amount = 0.001
epochs = 4000  # 20000  debug ****
n_classes = 2
if one_vector_flag:
    n_channels = 2
else:
    n_channels = 12

iterations_list = [10]
# iterations_list = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]   # debug ****
hier_list = ["procedure"] # "clinical_finding,"
negative_flags = ["hier", "area", "parea"] # []
c_list = [(x, y, z) for x in iterations_list for y in hier_list for z in negative_flags]

# global variables
# hier_name = "clinical_finding"
# hier_name = "procedure"
# directory_path = "/home/h/hl395/mlontology/SNO/"
directory_path = "SNO/"    # debug ****
img_path = directory_path + "img/"
#############################################################################################################
for iterations, hier_name, negative_flag in c_list:
# for iterations in iterations_list:
    print("Run training with hierarchy: {} iteration: {}".format(hier_name, iterations))
    print("Testing with negative taxonomy {}".format(negative_flag))

    vector_model_path = directory_path + "vectorModel/multiple/" + hier_name + "/" + str(iterations) + "/"
    data_path = directory_path + "data/" + hier_name + "/"

    # negative_flag = "area"  #
    if negative_flag == "hier":
        cnn_model_path = directory_path + "cnnModel/hier/" + vector_type + "/" + str(iterations) + "/"
        notPair_file = data_path + "taxNotPairs_sno_" + hier_name + "_hier.txt"
        img_name = hier_name + "_hier_" + vector_type + "_iter_" + str(iterations) + "_"
    elif negative_flag == "area":
        cnn_model_path = directory_path + "cnnModel/area/" + vector_type + "/" + str(iterations) + "/"
        notPair_file = data_path + "taxNotPairs_sno_" + hier_name + "_area.txt"
        img_name = hier_name + "_area_" + vector_type + "_iter_" + str(iterations) + "_"
    else:
        cnn_model_path = directory_path + "cnnModel/parea/" + vector_type + "/" + str(iterations) + "/"
        notPair_file = data_path + "taxNotPairs_sno_" + hier_name + "_parea.txt"
        img_name = hier_name + "_parea_" + vector_type + "_iter_" + str(iterations) + "_"

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

    #  PV-DBOW
    vector_model_file_0 = vector_model_path + "it_model0"
    pvdbow_it_model = load_vector_model(vector_model_file_0)

    # PV-DM seems better??
    vector_model_file_1 = vector_model_path + "it_model1"
    pvdm_it_model = load_vector_model(vector_model_file_1)

    #  PV-DBOW
    vector_model_file_0 = vector_model_path + "parent_model0"
    pvdbow_parent_model = load_vector_model(vector_model_file_0)

    # PV-DM seems better??
    vector_model_file_1 = vector_model_path + "parent_model1"
    pvdm_parent_model = load_vector_model(vector_model_file_1)

    #  PV-DBOW
    vector_model_file_0 = vector_model_path + "child_model0"
    pvdbow_child_model = load_vector_model(vector_model_file_0)

    # PV-DM seems better??
    vector_model_file_1 = vector_model_path + "child_model1"
    pvdm_child_model = load_vector_model(vector_model_file_1)

    vector_models_list = [pvdbow_it_model, pvdm_it_model, pvdbow_parent_model, pvdm_parent_model, pvdbow_child_model, pvdm_child_model ]

    # read both negative and positive pairs into pairs list and label list
    idpairs_list, label_list = readFromPairList(conceptPairList, conceptNotPairList)
    print(label_list[:20])

    # split samples into training and validation set
    from sklearn.model_selection import train_test_split

    X_train, X_validation, y_train, y_validation = train_test_split(idpairs_list, label_list, test_size=0.3,
                                                                    shuffle=True)
    print(X_train[:20])
    print(X_validation[:20])
    print(y_train[:20])
    print(y_validation[:20])


    # build the model
    graph = tf.Graph()

    # Construct placeholders
    with graph.as_default():
        inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name='inputs')
        labels_ = tf.placeholder(tf.float32, [None, n_classes], name='labels')
        keep_prob_ = tf.placeholder(tf.float32, name='keep')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

    with graph.as_default():
        # (batch, 512, 4) --> (batch, 256, 18)
        conv1 = tf.layers.conv1d(inputs=inputs_, filters=32, kernel_size=15, strides=1,
                                 padding='same', activation=tf.nn.leaky_relu)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

        # (batch, 256, 18) --> (batch, 128, 36)
        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=32, kernel_size=10, strides=1,
                                 padding='same', activation=tf.nn.leaky_relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

        # (batch, 128, 36) --> (batch, 64, 72)
        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=64, kernel_size=10, strides=1,
                                 padding='same', activation=tf.nn.leaky_relu)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

        # (batch, 64, 72) --> (batch, 32, 144)
        conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=64, kernel_size=10, strides=1,
                                 padding='same', activation=tf.nn.leaky_relu)
        max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

        # (batch, 32, 144) --> (batch, 16, 144)  # 288
        conv5 = tf.layers.conv1d(inputs=max_pool_4, filters=64, kernel_size=5, strides=1,
                                 padding='same', activation=tf.nn.leaky_relu)
        max_pool_5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2, padding='same')

        # (batch, 16, 144) --> (batch, 8, 144)   #576
        conv6 = tf.layers.conv1d(inputs=max_pool_5, filters=64, kernel_size=5, strides=1,
                                 padding='same', activation=tf.nn.leaky_relu)
        max_pool_6 = tf.layers.max_pooling1d(inputs=conv6, pool_size=2, strides=2, padding='same')

        # (batch, 16, 144) --> (batch, 8, 144)   #576
        # conv7 = tf.layers.conv1d(inputs=max_pool_5, filters=64, kernel_size=5, strides=1,
        #                          padding='same', activation=tf.nn.leaky_relu)
        # max_pool_7 = tf.layers.max_pooling1d(inputs=conv7, pool_size=2, strides=2, padding='same')

    with graph.as_default():
        # Flatten and add dropout
        flat = tf.reshape(max_pool_6, (-1, 8 * 64))
        flat = tf.layers.dense(flat, 200)

        flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

        # Predictions
        logits = tf.layers.dense(flat, n_classes, name='logits')
        logits_identity = tf.identity(input=logits, name="logits_identity")
        predict = tf.argmax(logits, 1, name="predict")  # the predicted class
        predict_identity = tf.identity(input=predict, name="predict_identity")
        probability = tf.nn.softmax(logits, name="probability")
        probability_identity = tf.identity(input=probability, name="probability_identity")

        # L2 loss prevents this overkill neural network to overfit the data
        l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

        # Cost function and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_)) + l2
        optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1), name='correct_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    # In[17]:

    if (os.path.exists(cnn_model_path) == False):
        os.makedirs(cnn_model_path)

    # In[ ]:

    validation_acc = []
    validation_loss = []

    train_acc = []
    train_loss = []
    # Best validation accuracy seen so far.
    best_validation_accuracy = 0.0

    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 0

        training_start_time = time.time()
        # Loop over epochs
        for e in range(epochs):

            # Loop over batches
            for x, y in get_batches_for_mulitple_vectors(X_train, y_train, one_vector_flag=one_vector_flag, conceptLabelDict=conceptLabelDict_2017, vector_models_list=vector_models_list, batch_size=batch_size):

                # Feed dictionary
                feed = {inputs_: x, labels_: y, keep_prob_: 0.5, learning_rate_: learning_rate}

                # Loss
                loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict=feed)
                train_acc.append(acc)
                train_loss.append(loss)

                # Print at each 50 iters
                if (iteration % 50 == 0):
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "Train loss: {:6f}".format(loss),
                          "Train acc: {:.6f}".format(acc)
                          )

                # Compute validation loss at every 100 iterations
                if (iteration % 100 == 0):
                    val_acc_ = []
                    val_loss_ = []

                    for x_v, y_v in get_batches_for_mulitple_vectors(X_validation, y_validation, one_vector_flag=one_vector_flag, conceptLabelDict= conceptLabelDict_2017, vector_models_list=vector_models_list, batch_size= batch_size):
                        # Feed
                        feed = {inputs_: x_v, labels_: y_v, keep_prob_: 1}

                        # Loss
                        loss_v, acc_v = sess.run([cost, accuracy], feed_dict=feed)
                        val_acc_.append(acc_v)
                        val_loss_.append(loss_v)

                    # If validation accuracy is an improvement over best-known.
                    acc_validation = np.mean(val_acc_)
                    if acc_validation > best_validation_accuracy:
                        # Update the best-known validation accuracy.
                        best_validation_accuracy = acc_validation
                        # Save all variables of the TensorFlow graph to file.
                        saver.save(sess=sess, save_path=cnn_model_path + "har.ckpt")
                        # A string to be printed below, shows improvement found.
                        improved_str = '*'
                    else:
                        # An empty string to be printed below.
                        # Shows that no improvement was found.
                        improved_str = ''

                    # Print info
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "Validation loss: {:6f}".format(np.mean(val_loss_)),
                          "Validation acc: {:.6f}".format(np.mean(val_acc_)),
                          "{}".format(improved_str*3))

                    # Store
                    validation_acc.append(np.mean(val_acc_))
                    validation_loss.append(np.mean(val_loss_))

                # Iterate
                iteration += 1
        training_duration = time.time() - training_start_time
        print("Total training time: {}".format(training_duration))
        # saver.save(sess, cnn_model_path + "har.ckpt")

    # In[ ]:



    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Plot training and test loss
    t = np.arange(iteration)

    plt.figure(figsize=(8, 6))
    plt.plot(t, np.array(train_loss), 'r-', t[t % 100 == 0][:len(validation_loss)], np.array(validation_loss), 'b*')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(img_path + img_name + 'loss.png')
    plt.show()

    # In[ ]:

    # Plot Accuracies
    plt.figure(figsize=(8, 6))
    plt.plot(t, np.array(train_acc), 'r-', t[t % 100 == 0][:len(validation_acc)], validation_acc, 'b*')
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    # plt.ylim(0.4, 1.0)
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(img_path + img_name + 'accuracy.png')
    plt.show()

    # In[ ]:

    print("result for testing leave-out samples: ")
    idpairs_list, label_list = readFromPairList([], testing_pair_lists)

    test_rest_acc = []
    batch_size = 4000

    n_classes = 2


    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint(cnn_model_path))
        test_iteration = 1

        for x_t, y_t in get_batches_for_mulitple_vectors(idpairs_list, label_list, one_vector_flag=one_vector_flag,
                                    conceptLabelDict=conceptLabelDict_2017, random_flag=False, vector_models_list= vector_models_list, batch_size=batch_size):
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1}

            batch_acc = sess.run(accuracy, feed_dict=feed)
            test_rest_acc.append(batch_acc)

            label_pred = sess.run(tf.argmax(logits, 1), feed_dict=feed)
            pred_prob = sess.run(tf.nn.softmax(logits), feed_dict=feed)

            test_iteration += 1
        print("Test accuracy: {:.6f}".format(np.mean(test_rest_acc)))

    # Now we're going to assess the quality of the neural net using ROC curve and AUC
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # send the actual dependent variable classifications for param 1,
    # and the confidences of the true classification for param 2.
    FPR, TPR, _ = roc_curve(label_list, pred_prob[:, 1])

    # Calculate the area under the confidence ROC curve.
    # This area is equated with the probability that the classifier will rank
    # a randomly selected defaulter higher than a randomly selected non-defaulter.
    AUC = auc(FPR, TPR)

    # What is "good" can dependm but an AUC of 0.7+ is generally regarded as good,
    # and 0.8+ is generally regarded as being excellent
    print("AUC is {}".format(AUC))

    # Now we'll plot the confidence ROC curve
    plt.figure()
    plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(img_path + img_name + 'roc.png')
    plt.show()

    from sklearn.metrics import classification_report

    print(classification_report(label_list, label_pred))
    plot_confusion_matrix(label_list, label_pred, img_path=img_path, img_name=img_name)

    # test new data
    print('\n\nTesting with new data')

    import json
    from pprint import pprint

    jsonFile = data_path + "2018" + hier_name + "_newconcepts.json"

    test_data = json.load(open(jsonFile))
    parents_dict = processConceptParents(test_data)
    all_existing_parents_set, _ = find_existing_parents_set_for_all_new_concepts(parents_dict, conceptLabelDict_2017)
    existing_parents_dict = find_concepts_with_existing_parents(parents_dict, conceptLabelDict_2017)
    positive_lists, negative_lists = prepare_samples_for_concepts_with_multiple_existing_parents(existing_parents_dict, all_existing_parents_set)

    # remove duplicates
    positive_lists = remove_duplicates(positive_lists)
    print("remove duplicates in positive test data, the number is: {}".format(len(positive_lists)))
    # remove duplicates
    negative_lists = remove_duplicates(negative_lists)
    print("remove duplicates in negative test data, the number is: {}".format(len(negative_lists)))
    # sampling data otherwise the negative testing sample is too huge
    positive_lists, negative_lists = sampling_data(positive_lists, negative_lists)
    data_list, label_list, borrowed_list = read_samples_into_data_label_borrowed(positive_lists, negative_lists)

    test_rest_acc = []
    true_label_list = []
    predicted_label_list = []
    predicted_prob = []
    batch_size = 4000

    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint(cnn_model_path))
        iteration = 0
        for x_t, y_t in sno_get_batches_for_testing_new_for_multiple_vectors(data_list, label_list, one_vector_flag=one_vector_flag,
                                    conceptLabelDict=conceptLabelDict_2018, vector_models_list=vector_models_list,
                                    batch_size=batch_size, random_flag=False, borrowed_list=borrowed_list):
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1}

            # true_label_list.extend(label_list)

            batch_acc = sess.run(accuracy, feed_dict=feed)
            test_rest_acc.append(batch_acc)

            # label_pred = sess.run(tf.argmax(logits, 1), feed_dict=feed)
            label_pred = sess.run(predict, feed_dict=feed)
            predicted_label_list.extend(label_pred)

            # pred_prob = sess.run(tf.nn.softmax(logits), feed_dict=feed)
            pred_prob = sess.run(probability, feed_dict=feed)
            predicted_prob.extend(pred_prob[:, 1])
            # print("\t\t Predict: ", label_pred)
            # print("\t\t True label: ", label_list)
            # Print at each 50 iters
            if (iteration % 5 == 0):
                print("Iteration: {:d}".format(iteration),
                      "batch acc: {:.6f}".format(batch_acc)
                      )
            iteration += 1

    print("Test accuracy: {:.6f}".format(np.mean(test_rest_acc)))

    from sklearn.metrics import classification_report

    true_label_list = label_list
    print(classification_report(true_label_list, predicted_label_list))

    # Now we're going to assess the quality of the neural net using ROC curve and AUC
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # send the actual dependent variable classifications for param 1,
    # and the confidences of the true classification for param 2.
    FPR, TPR, _ = roc_curve(true_label_list, predicted_prob)

    # Calculate the area under the confidence ROC curve.
    # This area is equated with the probability that the classifier will rank
    # a randomly selected defaulter higher than a randomly selected non-defaulter.
    AUC = auc(FPR, TPR)

    # What is "good" can dependm but an AUC of 0.7+ is generally regarded as good,
    # and 0.8+ is generally regarded as being excellent
    print("AUC is {}".format(AUC))

    # Now we'll plot the confidence ROC curve
    plt.figure()
    plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(img_path + img_name + 'roc2.png')
    plt.show()

    negative_sample_error_list = []
    for i, x in enumerate(true_label_list):
        if x != predicted_label_list[i] and x == 0:
            negative_sample_error_list.append(data_list[i])
    print(len(negative_sample_error_list))


    def get_label_from_id(id):
        if id in conceptLabelDict_2017:
            return conceptLabelDict_2017[id]
        elif id in conceptLabelDict_2018:
            return conceptLabelDict_2018[id]
        else:
            print("{} not exists in dictionary".format(id))


    # for batch_sample in negative_sample_error_list:
    #     print("{} : {} ".format(batch_sample[0], batch_sample[1]))
    #     print("{} -> {} ".format(get_label_from_id(batch_sample[0]), get_label_from_id(batch_sample[1])))
    sess.close()
    tf.reset_default_graph()
    print("One single test done \n\n")

print("testing done")