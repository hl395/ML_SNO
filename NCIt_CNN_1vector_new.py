# coding: utf-8
import tensorflow as tf
import os
import time
from ml_utils import *

# global setting
vector_type = "1vector"  # 2vectors
print("Processing {}".format(vector_type))
if vector_type == "1vector":
    one_vector_flag = True
else:
    one_vector_flag = False

###################################################################################
# set the model parameters
batch_size = 4000  # Batch size 4000 debug ****
seq_len = 128  # Number of steps
learning_rate = 0.0001
lambda_loss_amount = 0.001
epochs = 4000  # 4000 debug ****
n_classes = 2
if one_vector_flag:
    n_channels = 2
else:
    n_channels = 4

# iterations_list = [100]  # debug ****
iterations_list = [100, 200, 500, 1000]
negative_flags = ["hier", "area", "parea"]
combination_list = [(x, y) for x in iterations_list for y in negative_flags]

# global variables
# directory_path = "/home/h/hl395/mlontology/NCIt/"
directory_path = "NCIt/"   # debug ****
img_path = directory_path + "img/"
data_path = directory_path + "data/"
leaveout_file_path = data_path + "output.txt"
jsonFile = data_path + "18concepts.json"
#####################################################################################
for iterations, negative_flag in combination_list:
    print("Testing with negative taxonomy {}".format(negative_flag))
    print('Run training with iter: ', str(iterations))

    vector_model_path = directory_path + "vectorModel/" + str(iterations) + "/"

    # negative_flag = "area"  #
    if negative_flag == "hier":
        cnn_model_path = directory_path + "cnnModel/hier/" + vector_type + "/" + str(iterations) + "/"
        notPair_file = data_path + "taxNotPairs_owl_ncit_hier.txt"
        img_name = "hier_" + vector_type + "_iter_" + str(iterations) + "_"
    elif negative_flag == "area":
        cnn_model_path = directory_path + "cnnModel/area/" + vector_type + "/" + str(iterations) + "/"
        notPair_file = data_path + "taxNotPairs_owl_ncit_area.txt"
        img_name = "area_" + vector_type + "_iter_" + str(iterations) + "_"
    else:
        cnn_model_path = directory_path + "cnnModel/parea/" + vector_type + "/" + str(iterations) + "/"
        notPair_file = data_path + "taxNotPairs_owl_ncit_parea.txt"
        img_name = "parea_" + vector_type + "_iter_" + str(iterations) + "_"

    positive_flag = "hier"  #
    if positive_flag == "hier":
        pair_file = data_path + "taxPairs_owl_ncit_hier.txt"
    elif positive_flag == "area":
        pair_file = data_path + "taxPairs_owl_ncit_area.txt"
    else:
        pair_file = data_path + "taxPairs_owl_ncit_parea.txt"

    print("Positive training data from {}".format(positive_flag))
    print("Negative training data from {}".format(negative_flag))

    # read class label file
    # create mapping from id to labels
    # iso-8859-1 , encoding="iso-8859-1"
    label_file = data_path + "ontClassLabels_owl_ncit.txt"
    conceptLabelDict, errors = read_label(label_file)
    print(conceptLabelDict["4863"])
    print(conceptLabelDict["115117"])
    print(errors)

    # read from a file with the randomly chosen 2000 positive and 2000 negative samples for testing
    print('Leave out for testing')
    testing_pair_lists = read_leaveout_from_file(leaveout_file_path)
    print(len(testing_pair_lists))
    print(testing_pair_lists[:5])

    # read positive samples
    conceptPairList, _ = read_pair(pair_file)
    check_pairs = conceptPairList[10:15]
    print(check_pairs)
    print(len(conceptPairList))

    # read negative samples
    conceptNotPairList, _ = read_not_pair(notPair_file)
    check_pairs = conceptNotPairList[10:15]
    print(check_pairs)
    print(len(conceptNotPairList))

    # remove duplicates
    print("remove duplicates")
    conceptPairList = remove_duplicates(conceptPairList)
    print("After remove duplicates in linked pairs: ")
    print(len(conceptPairList))
    conceptNotPairList = remove_duplicates(conceptNotPairList)
    print("After remove duplicates in not linked pairs: ")
    print(len(conceptNotPairList))

    # remove testing pairs from training
    testing_list = [x for x in conceptPairList if x not in testing_pair_lists]
    testing_list_2 = [y for y in conceptNotPairList if y not in testing_pair_lists]

    print("After remove testing pairs from linked pairs: ")
    print(len(testing_list))
    print("After remove testing pairs from not linked pairs: ")
    print(len(testing_list_2))

    conceptPairList = testing_list
    conceptNotPairList = testing_list_2

    # In[8]:
    conceptPairList, conceptNotPairList = sampling_data(conceptPairList, conceptNotPairList)

    #  PV-DBOW
    vector_model_file_0 = vector_model_path + "model0"
    pvdbow_model = load_vector_model(vector_model_file_0)

    # PV-DM seems better??
    vector_model_file_1 = vector_model_path + "model1"
    pvdm_model = load_vector_model(vector_model_file_1)
    pvdm_model.docvecs['7918']

    # read both negative and positive pairs into pairs list and label list
    idpairs_list, label_list = readFromPairList(conceptPairList, conceptNotPairList)
    print(label_list[:20])

    test_string = '''4092,3677,1
        4556,4555,1
        4408,4242,1
        62210,4133,1
        4459,84509,1
        6750,2896,1
        3942,7158,1
        3754,65157,1
        3084,65157,1
        6985,65157,1
        3061,65157,1
        27939,65157,1
        4139,2915,1
        3058,2937,1
        4047,2937,1
        7335,3010,1
        3375,9343,1
        4150,45921,1'''

    X_test, y_test = read_sample_from_string(test_string)

    print(len(X_test))
    print(X_test)
    print(y_test)

    neg_test_string = '''4092,6971,0
            3061,5329,0
            4150,4068,0
            3084,3723,0
            6750,4223,0
            3375,7337,0
            62210,40384,0
            6985,40133,0
            3058,35427,0
            4047,5419,0
            7335,3077,0
            27939,3077,0
            4556,4364,0
            4459,65176,0
            3942,4634,0
            3754,6930,0
            4139,3784,0
            4408,6062,0'''
    negative_data_list, negative_label_list = read_sample_from_string(neg_test_string)

    print(len(negative_data_list))
    print(negative_data_list)
    print(negative_label_list)

    # remove positive testing samples from training
    idpairs_list, label_list = remove_testing_samples_from_training(X_test, idpairs_list, label_list)
    # remove the negative non is-a testing samples if already exist in positive testing samples
    negative_data_list, negative_label_list = remove_conflicts(negative_data_list, negative_label_list, X_test)
    # remove negative testing samples from training
    idpairs_list, label_list = remove_testing_samples_from_training(negative_data_list, idpairs_list, label_list)

    # read the negative testing samples for the above 18 samples
    test_data = json.load(open(jsonFile))
    # read all uncles into list
    paired_list, uncle_not_paired_list = read18FromJsonData(test_data)
    uncle_not_paired_list = remove_duplicates(uncle_not_paired_list)
    # # read all siblings into list
    # sib_not_paired_list = read18SiblingsFromJsonData(test_data)
    # sib_not_paired_list = remove_duplicates(sib_not_paired_list)
    # read uncle list to data label pairs
    n_data_list, n_label_list = readFromPairList([], uncle_not_paired_list)
    # # read sibling list to data label pairs
    # sib_data_list, sib_label_list = readFromPairList([], sib_not_paired_list)
    # remove the negative sibling testing samples if already exist in positive testing samples
    n_data_list, n_label_list = remove_conflicts(n_data_list, n_label_list, X_test)
    # remove the negative sibling testing samples if already exist in negative testing samples
    n_data_list, n_label_list = remove_conflicts(n_data_list, n_label_list, negative_data_list)
    # append those not chosen negative samples to training data set
    idpairs_list, label_list = append_list(n_data_list, n_label_list, idpairs_list, label_list)

    # remove the negative sibling testing samples if already exist in positive testing samples
    # sib_data_list, sib_label_list = remove_conflicts(sib_data_list, sib_label_list, X_test)
    # randomly pick one uncle for each of the above 18 concepts if their uncles exist,
    # if no uncles randomly pick one sibling
    # negative_data_list, negative_label_list, n_data_list, n_label_list = prepare_negative_testing_data(n_data_list,
    #                                                             n_label_list, sib_data_list, sib_label_list, X_test)

    print("final training size is ")
    from scipy.stats import itemfreq
    freq = itemfreq(label_list)
    print(freq[:, 0])
    print(freq[:, 1])

    # split samples into training and validation set
    # In[12]:

    from sklearn.model_selection import train_test_split

    X_train, X_validation, y_train, y_validation = train_test_split(idpairs_list, label_list, test_size=0.2,
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
        # (batch, 128, 2) --> (batch, 64, 18)
        conv1 = tf.layers.conv1d(inputs=inputs_, filters=18, kernel_size=2, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

        # (batch, 64, 18) --> (batch, 32, 36)
        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

        # (batch, 32, 36) --> (batch, 16, 72)
        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

        # (batch, 16, 72) --> (batch, 8, 144)
        conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=2, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

    with graph.as_default():
        # Flatten and add dropout
        flat = tf.reshape(max_pool_4, (-1, 8 * 144))
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
            for x, y in get_batches(X_train, y_train, one_vector_flag=one_vector_flag,
                                    conceptLabelDict=conceptLabelDict, pvdm_model=pvdm_model,
                                    pvdbow_model=pvdbow_model, batch_size=batch_size):

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

                    for x_v, y_v in get_batches(X_validation, y_validation, one_vector_flag=one_vector_flag,
                                                conceptLabelDict=conceptLabelDict,
                                                pvdbow_model=pvdbow_model, pvdm_model=pvdm_model,
                                                batch_size=batch_size):
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
                          " {}".format(improved_str * 3))

                    # Store
                    validation_acc.append(np.mean(val_acc_))
                    validation_loss.append(np.mean(val_loss_))

                # Iterate
                iteration += 1
        training_duration = time.time() - training_start_time
        print("Total training time: {}".format(training_duration))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Plot training and test loss
    t = np.arange(iteration)

    plt.figure(figsize=(8, 6))
    plt.plot(t, np.array(train_loss), 'r-', t[t % 100 == 0], np.array(validation_loss), 'b*')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(img_path + img_name + 'loss.png')
    plt.show()

    # Plot Accuracies
    plt.figure(figsize=(8, 6))
    plt.plot(t, np.array(train_acc), 'r-', t[t % 100 == 0], validation_acc, 'b*')
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.ylim(0.4, 1.0)
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(img_path + img_name + 'accuracy.png')
    plt.show()

    print("{}".format("#" * 20))
    print("testing positive {} concept pairs".format(len(X_test)))
    test_acc = []
    batch_size = len(X_test)
    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint(cnn_model_path))

        for x_t, y_t in get_batches(X_test, y_test, one_vector_flag=one_vector_flag, conceptLabelDict=conceptLabelDict,
                                    pvdbow_model=pvdbow_model,
                                    pvdm_model=pvdm_model, batch_size=batch_size):
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1}

            batch_acc = sess.run(accuracy, feed_dict=feed)
            test_acc.append(batch_acc)
        print("Test accuracy: {:.6f}".format(np.mean(test_acc)))

    print("{}".format("#" * 20))
    print("testing negative {} concept pairs".format(len(negative_data_list)))
    test_acc = []
    batch_size = len(negative_data_list)
    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint(cnn_model_path))

        for x_t, y_t in get_batches(negative_data_list, negative_label_list, one_vector_flag=one_vector_flag,
                                    conceptLabelDict=conceptLabelDict, pvdbow_model=pvdbow_model,
                                    pvdm_model=pvdm_model, batch_size=batch_size):
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1}

            batch_acc = sess.run(accuracy, feed_dict=feed)
            test_acc.append(batch_acc)
        print("Negative test accuracy: {:.6f}".format(np.mean(test_acc)))


    testingPair_file = data_path + "testing_owl_ncit.txt"
    testingPairList, _ = read_test_pair(testingPair_file, model=pvdm_model)
    check_pairs = testingPairList[10:15]
    print(check_pairs)
    print(len(testingPairList))

    # remove samples from training samples
    testingPairList = from_testing_remove_training_samples(conceptPairList, conceptNotPairList, testingPairList)

    print("preparing done, start to read to pairs and labels:")
    # testing
    idpairs_list, label_list = readFromPairList([], testingPairList)
    print("reading done, start to testing:")

    test_rest_acc = []
    batch_size = 5000
    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint(cnn_model_path))
        test_iteration = 0

        for x_t, y_t in get_batches(idpairs_list, label_list, one_vector_flag=one_vector_flag,
                                    conceptLabelDict=conceptLabelDict,
                                    pvdbow_model=pvdbow_model, pvdm_model=pvdm_model, batch_size=batch_size):
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1}

            batch_acc = sess.run(accuracy, feed_dict=feed)
            test_rest_acc.append(batch_acc)
            if (test_iteration % 50 == 0):
                print("Iteration: {} ".format(test_iteration), "Accuracy: {:.6f}".format(batch_acc))
            test_iteration += 1
        print("Test accuracy: {:.6f}".format(np.mean(test_rest_acc)))


    def get_batches_withName(x_samples, y_samples, batch_size=64, n_classes=2):
        samples = list(zip(x_samples, y_samples))
        num_samples = len(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            X_samples = []
            Y_samples = []
            for batch_sample in batch_samples:
                print("{} : {} ".format(batch_sample[0], batch_sample[1]))
                print("{} -> {} ".format(conceptLabelDict[batch_sample[0][0]], conceptLabelDict[batch_sample[0][1]]))
                pair_list = batch_sample[0]
                #            data_vector= getVector(pair_list, conceptLabelDict, vector_model)
                pvdm_vector = getVector(pair_list, conceptLabelDict, pvdm_model)
                pvdbow_vector = getVector(pair_list, conceptLabelDict, pvdbow_model)
                data_vector = stack_vector(pvdm_vector=pvdm_vector, pvdbow_vector=pvdbow_vector,
                                           one_vector_flag=one_vector_flag)
                #             print(data_vector.shape)
                X_samples.append(data_vector)
                class_label = batch_sample[1]
                Y_samples.append(class_label)

            X_samples = np.array(X_samples).astype('float32')
            Y_samples = np.eye(n_classes)[Y_samples]
            #             print('one batch ready')
            yield X_samples, Y_samples


    test_pred = []
    batch_size = len(X_test)
    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint(cnn_model_path))

        for x_t, y_t in get_batches_withName(X_test, y_test, batch_size=batch_size):
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1}

            batch_pred = sess.run(correct_pred, feed_dict=feed)
            print("Test result: {}".format(batch_pred))
            test_pred.append(batch_pred)

    print(test_pred)

    test_pred = []
    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint(cnn_model_path))

        for pair, label in zip(X_test, y_test):
            x_t, y_t = get_one_vector_batch(pair, label, conceptLabelDict, one_vector_flag=one_vector_flag,
                                            pvdm_model=pvdm_model, pvdbow_model=pvdbow_model)
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1}

            one_pred = sess.run(correct_pred, feed_dict=feed)
            print("One by one test result: {}".format(one_pred))
            test_pred.append(one_pred)

    print(test_pred)

    print("now testing the leave out samples ")
    idpairs_list, label_list = readFromPairList([], testing_pair_lists)

    test_rest_acc = []
    batch_size = 4000
    n_classes = 2

    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint(cnn_model_path))
        test_iteration = 1

        for x_t, y_t in get_batches(idpairs_list, label_list, one_vector_flag=one_vector_flag, batch_size=batch_size,
                                    n_classes=n_classes,
                                    random_flag=False, conceptLabelDict=conceptLabelDict, pvdbow_model=pvdbow_model,
                                    pvdm_model=pvdm_model):
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
    plot_confusion_matrix(label_list, label_pred, img_path, img_name)

    negative_sample_error_list = []
    for i, x in enumerate(label_list):
        if x != label_pred[i] and x == 0:
            negative_sample_error_list.append(idpairs_list[i])
    print(len(negative_sample_error_list))

    for batch_sample in negative_sample_error_list:
        print("{} : {} ".format(batch_sample[0], batch_sample[1]))
        print("{} -> {} ".format(conceptLabelDict[batch_sample[0]], conceptLabelDict[batch_sample[1]]))

    sess.close()
    tf.reset_default_graph()
    print("One single test done \n\n")
print("testing done")