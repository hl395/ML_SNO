# coding: utf-8
import os
import time
import tensorflow as tf
import matplotlib
from ml_utils import *
import csv
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("data_dir", None, "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")
flags.DEFINE_string("vocab_file", None,"The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("output_dir", None, "The output directory where the model checkpoints will be written.")
flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_string("directory_path", None, "directory path")
flags.DEFINE_string("vector_type", '2vectors', "how many vectors per concept.")
flags.DEFINE_integer("n_channels", 4, "how many channels to the CNN model.")
flags.DEFINE_integer("batch_size", 4000, "how many samples per batch.")
flags.DEFINE_integer("seq_len", 256, "embedding size for each concept.")
flags.DEFINE_float("learning_rate", 0.0001, "learning rate for the CNN model.")
flags.DEFINE_bool("l2_loss_flag", False, "l2 loss flag.")
flags.DEFINE_float("lambda_loss_amount", 0.001, "l2 loss.")
flags.DEFINE_integer("epochs", 10,  "200  debug ****")
flags.DEFINE_integer("n_classes", 2, "how many channels to the CNN model.")
flags.DEFINE_string("img_path", "img/", "")
flags.DEFINE_bool("one_vector_flag", False, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_integer("iterations", 40, "vector training parameters: 10, 20, 40.")
flags.DEFINE_string("hier_name", None, "hierarchy name: clinical_finding, procedure")
flags.DEFINE_string("taxonomy", "hier", "summarization type: hier, area, parea")
flags.DEFINE_string("vector_model_path", None, "where to get vector embeddings")
flags.DEFINE_string("img_name", "img__" + timestamp, "where to get vector embeddings")
flags.DEFINE_string("task_name", "CNN", "task name")
######################################################################################################

######################################################################################################

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    print("Run training with hierarchy: {} iteration: {}".format(FLAGS.hier_name, FLAGS.iterations))
    print("Testing with negative taxonomy {}".format(FLAGS.taxonomy))

    processors = {
      "cnn": CNNProcessor
    }
    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    #  PV-DBOW
    vector_model_file_0 = FLAGS.vector_model_path + "model0"
    pvdbow_model = gensim.models.Doc2Vec.load(vector_model_file_0)

    # PV-DM seems better??
    vector_model_file_1 = FLAGS.vector_model_path + "model1"
    pvdm_model = gensim.models.Doc2Vec.load(vector_model_file_1)

    train_examples = None
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir, FLAGS.hier_name, FLAGS.taxonomy)
        num_actual_train_examples = len(train_examples)
        print("number of {} examples: {}".format("train", num_actual_train_examples))
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir, FLAGS.hier_name, FLAGS.taxonomy)
        num_actual_eval_examples = len(eval_examples)
        print("number of {} examples: {}".format("eval", num_actual_eval_examples))
    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir, FLAGS.hier_name, FLAGS.taxonomy)
        num_actual_predict_examples = len(predict_examples)
        print("number of {} examples: {}".format("predict", num_actual_predict_examples))
    # for (ex_index, example) in enumerate(train_examples):
    #     print("id_a:{}\ttext_a: {}".format(example.id_a, example.text_a))
    #     print("id_b:{}\ttext_b: {}".format(example.id_b, example.text_b))
    # split samples into training and validation set
    from sklearn.model_selection import train_test_split
    train_examples, validation_examples = train_test_split(train_examples, test_size=0.1, shuffle=True)

    # build the model
    graph = tf.Graph()
    # Construct placeholders
    with graph.as_default():
        inputs_ = tf.placeholder(tf.float32, [None, FLAGS.seq_len, FLAGS.n_channels], name='inputs')
        labels_ = tf.placeholder(tf.float32, [None, FLAGS.n_classes], name='labels')
        keep_prob_ = tf.placeholder(tf.float32, name='keep')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

    with graph.as_default():
        # (batch, 512, 4) --> (batch, 256, 18)
        conv1 = tf.layers.conv1d(inputs=inputs_, filters=32, kernel_size=15, strides=1,
                                 padding='same', activation=tf.nn.leaky_relu, dilation_rate=1)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

        # (batch, 256, 18) --> (batch, 128, 36)
        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=32, kernel_size=10, strides=1,
                                 padding='same', activation=tf.nn.leaky_relu, dilation_rate=2)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

        # (batch, 128, 36) --> (batch, 64, 72)
        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=64, kernel_size=10, strides=1,
                                 padding='same', activation=tf.nn.leaky_relu, dilation_rate=3)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

        # (batch, 64, 72) --> (batch, 32, 144)
        conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=64, kernel_size=10, strides=1,
                                 padding='same', activation=tf.nn.leaky_relu, dilation_rate=4)
        max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

        # (batch, 32, 144) --> (batch, 16, 144)  # 288
        conv5 = tf.layers.conv1d(inputs=max_pool_4, filters=64, kernel_size=5, strides=1,
                                 padding='same', activation=tf.nn.leaky_relu, dilation_rate=5)
        max_pool_5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2, padding='same')

        # (batch, 16, 144) --> (batch, 8, 144)   #576
        conv6 = tf.layers.conv1d(inputs=max_pool_5, filters=64, kernel_size=5, strides=1,
                                 padding='same', activation=tf.nn.leaky_relu, dilation_rate=6)
        max_pool_6 = tf.layers.max_pooling1d(inputs=conv6, pool_size=2, strides=2, padding='same')

    with graph.as_default():
        # Flatten and add dropout
        flat = tf.reshape(max_pool_5, (-1, 8 * 64))
        flat = tf.layers.dense(flat, 200)

        flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

        # Predictions
        logits = tf.layers.dense(flat, FLAGS.n_classes, name='logits')
        logits_identity = tf.identity(input=logits, name="logits_identity")
        predict = tf.argmax(logits, 1, name="predict")  # the predicted class
        predict_identity = tf.identity(input=predict, name="predict_identity")
        probability = tf.nn.softmax(logits, name="probability")
        probability_identity = tf.identity(input=probability, name="probability_identity")

        # L2 loss prevents this overkill neural network to overfit the data
        l2 = FLAGS.lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        if FLAGS.l2_loss_flag:
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_)) + l2
        # Cost function and optimizer
        else:
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
        optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1), name='correct_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    # In[17]:

    if (os.path.exists(FLAGS.output_dir) == False):
        os.makedirs(FLAGS.output_dir)

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
        for e in range(FLAGS.epochs):

            # Loop over batches
            for x, y in get_batches(train_examples, tokenizer=tokenizer, pvdm_model=pvdm_model, pvdbow_model=pvdbow_model, batch_size=FLAGS.batch_size):

                # Feed dictionary
                feed = {inputs_: x, labels_: y, keep_prob_: 0.5, learning_rate_: FLAGS.learning_rate}

                # Loss
                loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict=feed)
                train_acc.append(acc)
                train_loss.append(loss)

                # Print at each 50 iters
                if (iteration % 50 == 0):
                    print("Epoch: {}/{}".format(e, FLAGS.epochs),
                          "Iteration: {:d}".format(iteration),
                          "Train loss: {:6f}".format(loss),
                          "Train acc: {:.6f}".format(acc)
                          )

                # Compute validation loss at every 100 iterations
                if (iteration % 100 == 0):
                    val_acc_ = []
                    val_loss_ = []

                    for x_v, y_v in get_batches(validation_examples, tokenizer=tokenizer, pvdm_model=pvdm_model, pvdbow_model=pvdbow_model, batch_size=FLAGS.batch_size):
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
                        # saver.save(sess=sess, save_path=cnn_model_path + "har.ckpt")
                        # A string to be printed below, shows improvement found.
                        improved_str = '*'
                    else:
                        # An empty string to be printed below.
                        # Shows that no improvement was found.
                        improved_str = ''

                    # Print info
                    print("Epoch: {}/{}".format(e, FLAGS.epochs),
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
        saver.save(sess, FLAGS.output_dir + "har.ckpt")

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Plot training and test loss
    t = np.arange(iteration)

    plt.figure(figsize=(8, 6))
    plt.plot(t, np.array(train_loss), 'r-', t[t % 100 == 0][:len(validation_loss)], np.array(validation_loss), 'b*')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(FLAGS.img_path + FLAGS.img_name + 'training_loss.png')
    plt.show()

    # In[ ]:

    # Plot Accuracies
    plt.figure(figsize=(8, 6))
    plt.plot(t, np.array(train_acc), 'r-', t[t % 100 == 0][:len(validation_acc)], validation_acc, 'b*')
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    # plt.ylim(0.4, 1.0)
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(FLAGS.img_path + FLAGS.img_name + 'training_accuracy.png')
    plt.show()

    # test new data
    tf.logging.info("***** Running evaluation*****")
    tf.logging.info("  Num examples = {}".format(num_actual_eval_examples))

    eval_acc = []
    eval_label_pred = []
    eval_pred_prob = []
    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.output_dir))
        test_iteration = 1

        for x_t, y_t in get_batches(eval_examples, tokenizer=tokenizer, pvdm_model=pvdm_model,
                                    pvdbow_model=pvdbow_model, batch_size=FLAGS.batch_size, random_flag=False):
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1}

            batch_acc = sess.run(accuracy, feed_dict=feed)
            eval_acc.append(batch_acc)

            eval_label_pred.extend(sess.run(tf.argmax(logits, 1), feed_dict=feed))
            eval_pred_prob.extend(sess.run(tf.nn.softmax(logits), feed_dict=feed))

            test_iteration += 1
        print("Test accuracy: {:.6f}".format(np.mean(eval_acc)))

    # Now we're going to assess the quality of the neural net using ROC curve and AUC
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # send the actual dependent variable classifications for param 1,
    # and the confidences of the true classification for param 2.
    eval_true_labels = [x.label for x in eval_examples]

    FPR, TPR, _ = roc_curve(eval_true_labels, column(eval_pred_prob, 1))

    # Calculate the area under the confidence ROC curve.
    # This area is equated with the probability that the classifier will rank
    # a randomly selected defaulter higher than a randomly selected non-defaulter.
    AUC = auc(FPR, TPR)

    # What is "good" can dependm but an AUC of 0.7+ is generally regarded as good,
    # and 0.8+ is generally regarded as being excellent
    print("AUC is {}".format(AUC))

    # Now we'll plot the confidence ROC curve
    plt.figure()
    plt.plot(FPR, TPR, label='EVA ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(FLAGS.img_path + FLAGS.img_name + 'eva_roc.png')
    plt.show()

    from sklearn.metrics import classification_report
    tf.logging.info("***** Evaluation result *****")
    # tf.logging.info('true label size: {}\tlabel_pred size: {}'.format(len(eval_true_labels), len(eval_label_pred)))
    tf.logging.info(classification_report(eval_true_labels, eval_label_pred))
    plot_confusion_matrix(eval_true_labels, eval_label_pred, img_path=FLAGS.img_path, img_name=FLAGS.img_name+"eva_")



    # test new data
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = {}".format(num_actual_predict_examples))

    test_rest_acc =[]
    label_pred = []
    pred_prob = []
    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.output_dir))
        test_iteration = 1

        for x_t, y_t in get_batches(predict_examples, tokenizer=tokenizer, pvdm_model=pvdm_model,
                                    pvdbow_model=pvdbow_model, batch_size=FLAGS.batch_size, random_flag=False):
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1}

            batch_acc = sess.run(accuracy, feed_dict=feed)
            test_rest_acc.append(batch_acc)

            label_pred.extend(sess.run(tf.argmax(logits, 1), feed_dict=feed))
            pred_prob.extend(sess.run(tf.nn.softmax(logits), feed_dict=feed))

            test_iteration += 1
        print("Test accuracy: {:.6f}".format(np.mean(test_rest_acc)))

    # Now we're going to assess the quality of the neural net using ROC curve and AUC
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # send the actual dependent variable classifications for param 1,
    # and the confidences of the true classification for param 2.
    true_labels = [x.label for x in predict_examples]

    FPR, TPR, _ = roc_curve(true_labels, column(pred_prob, 1))

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
    plt.savefig(FLAGS.img_path + FLAGS.img_name + 'predict_roc.png')
    plt.show()

    from sklearn.metrics import classification_report
    tf.logging.info("***** Prediction result *****")
    # pprint("true label size: {}\t label_pred size: {}".format(len(true_labels), len(label_pred)))
    tf.logging.info(classification_report(true_labels, label_pred))
    plot_confusion_matrix(true_labels, label_pred, img_path=FLAGS.img_path, img_name=FLAGS.img_name+"predict_")


    sess.close()
    tf.reset_default_graph()
    print("Test done \n\n")



if __name__ == "__main__":
  # flags.mark_flag_as_required("data_dir")
  # flags.mark_flag_as_required("vocab_file")
  # flags.mark_flag_as_required("output_dir")

  FLAGS.directory_path = "SNO/"
  FLAGS.iterations = 40
  FLAGS.seq_len = 256
  FLAGS.hier_name = "clinical_finding"
  FLAGS.taxonomy = "hier"
  FLAGS.vector_model_path = FLAGS.directory_path + "vectorModel/" + FLAGS.hier_name + "/" + str(FLAGS.iterations) + "/"
  FLAGS.img_path = FLAGS.directory_path +"img/"
  FLAGS.img_name = timestamp + FLAGS.hier_name + "_" + FLAGS.taxonomy + "_iter_" + str(FLAGS.iterations) + "_dilated_"
  FLAGS.output_dir = FLAGS.directory_path + "cnnModel/" + FLAGS.hier_name + "_" + str(FLAGS.iterations) + "/dilated/"
  FLAGS.vocab_file = "MODEL/small/vocab.txt"
  FLAGS.batch_size = 4000
  FLAGS.epochs = 200
  # FLAGS.data_dir = "../bert/glue_data/MYTESTING/Iteration_1/"
  FLAGS.data_dir = FLAGS.directory_path + "TestData/" + "Iteration_1/clinical_finding/"

  # FLAGS.init_checkpoint = "tmp/procedure/pretraining_output/model.ckpt-10000"  # load the pre-trained model


  FLAGS.do_train = True  # True
  FLAGS.do_eval = True  # True
  FLAGS.do_predict = True  # False
  FLAGS.task_name = "CNN"

  tf.app.run()
