import os
import re
import smart_open
import random
import gensim
from pprint import pprint
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.utils import shuffle
import csv
import tokenization
import tensorflow as tf

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, id_a, id_b, text_a, text_b, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.id_a = id_a
    self.id_b = id_b
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class CNNProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  def get_train_examples(self, data_dir, hier, tax):
    """See base class."""
    file_name = hier + "_" + tax + "_"
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, file_name+"train.tsv")), "train")

  def get_dev_examples(self, data_dir, hier, tax):
    """See base class."""
    file_name = hier + "_" + tax + "_"
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, file_name+"dev.tsv")), "dev")

  def get_test_examples(self, data_dir, hier, tax):
    """See base class."""
    file_name = hier + "_" + tax + "_"
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, file_name+"test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      id_a = int(tokenization.convert_to_unicode(line[1]))
      id_b = int(tokenization.convert_to_unicode(line[2]))
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = int(tokenization.convert_to_unicode(line[-1]))
      else:
        label = int(tokenization.convert_to_unicode(line[0]))
      examples.append(
          InputExample(guid=guid, id_a=id_a, id_b=id_b, text_a=text_a, text_b=text_b, label=label))
    return examples


def get_batches(train_examples, tokenizer, pvdm_model, pvdbow_model, batch_size=64, n_classes=2, random_flag=True):
    samples = train_examples
    num_samples = len(samples)
    if random_flag:
        shuffle(samples)
    for offset in range(0, num_samples, batch_size):
        batch_samples = samples[offset:offset + batch_size]

        X_samples = []
        Y_samples = []
        for batch_sample in batch_samples:
            id_pair_list = [batch_sample.id_a, batch_sample.id_b]
            text_pair_list = [batch_sample.text_a, batch_sample.text_b]
            # data_vector = getVector(pair_list, conceptLabelDict, vector_model)
            pvdm_vector = getVector(id_pair_list, text_pair_list, pvdm_model, tokenizer)
            pvdbow_vector = getVector(id_pair_list, text_pair_list, pvdbow_model, tokenizer)
            data_vector = stack_vector(pvdm_vector=pvdm_vector, pvdbow_vector=pvdbow_vector)
            # data_vector = stackVector(data_vector)
            # print(data_vector.shape)
            X_samples.append(data_vector)
            class_label = batch_sample.label
            Y_samples.append(class_label)

        X_samples = np.array(X_samples).astype('float32')
        Y_samples = np.eye(n_classes)[Y_samples]
        #             print('one batch ready')
        if random_flag:
            yield shuffle(X_samples, Y_samples)
        else:
            yield (X_samples, Y_samples)

def get_batches_elmo(train_examples, embed, batch_size=64, n_classes=2, random_flag=True):
    samples = train_examples
    num_samples = len(samples)
    if random_flag:
        shuffle(samples)
    for offset in range(0, num_samples, batch_size):
        batch_samples = samples[offset:offset + batch_size]
        X_samples = []
        Y_samples = []
        for batch_sample in batch_samples:
            id_pair_list = [batch_sample.id_a, batch_sample.id_b]
            a_vector = get_vector_from_elmo(embed, batch_sample.id_a)
            b_vector = get_vector_from_elmo(embed, batch_sample.id_b)
            data_vector = np.array((a_vector, b_vector)).T
            X_samples.append(data_vector)
            class_label = batch_sample.label
            Y_samples.append(class_label)

        X_samples = np.array(X_samples).astype('float32')
        Y_samples = np.eye(n_classes)[Y_samples]
        #             print('one batch ready')
        assert len(X_samples) == len(Y_samples)
        if random_flag:
            yield shuffle(X_samples, Y_samples)
        else:
            yield (X_samples, Y_samples)

def get_batches_elmo_multi(train_examples, embed, embed_2, batch_size=64, n_classes=2, random_flag=True):
    samples = train_examples
    num_samples = len(samples)
    if random_flag:
        shuffle(samples)
    for offset in range(0, num_samples, batch_size):
        batch_samples = samples[offset:offset + batch_size]
        X_samples = []
        Y_samples = []
        for batch_sample in batch_samples:
            id_pair_list = [batch_sample.id_a, batch_sample.id_b]
            a_vector = get_vector_from_elmo(embed, batch_sample.id_a)
            b_vector = get_vector_from_elmo(embed, batch_sample.id_b)
            a_vector_2 = get_vector_from_elmo(embed_2, batch_sample.id_a)
            b_vector_2 = get_vector_from_elmo(embed_2, batch_sample.id_b)
            c_vector = np.concatenate((a_vector_2, b_vector_2), axis=0)
            data_vector = np.array((a_vector, b_vector, c_vector)).T
            X_samples.append(data_vector)
            class_label = batch_sample.label
            Y_samples.append(class_label)

        X_samples = np.array(X_samples).astype('float32')
        Y_samples = np.eye(n_classes)[Y_samples]
        #             print('one batch ready')
        assert len(X_samples) == len(Y_samples)
        if random_flag:
            yield shuffle(X_samples, Y_samples)
        else:
            yield (X_samples, Y_samples)

def getVectorFromModel(concept_id, concept_label, model, tokenizer):
    if concept_id in model.docvecs:
        concept_vector = model.docvecs[concept_id]
    else:
        concept_vector = model.infer_vector(tokenizer.tokenize(concept_label))
    return concept_vector

def getVector(pair_list, text_pair_list, model, tokenizer):
    a = getVectorFromModel(pair_list[0], text_pair_list[0], model, tokenizer)
    b = getVectorFromModel(pair_list[1], text_pair_list[1], model, tokenizer)
    c = np.array((a, b))
    c = c.T
    #     c = np.expand_dims(c, axis=2)
    #     print(c.shape)
    return c

# stack vectors into 4 channels

def stack_vector(pvdm_vector, pvdbow_vector):
    return np.concatenate((pvdm_vector, pvdbow_vector), axis=1)


def get_vector_from_elmo(embed, key):
    if key in embed:
        return embed[key]
    else:
        print("embeddings for concept id {} does not exist".format(key))

def get_embeddings_from_elmo(examples, batch_size=100):
    # Create graph and finalize (optional but recommended).
    concept_dict = {}
    for example in examples:
        if example.id_a not in concept_dict:
            concept_dict[example.id_a] = example.text_a
        if example.id_b not in concept_dict:
            concept_dict[example.id_b] = example.text_b

    id_list = []
    text_list = []
    for i, text in concept_dict.items():
        id_list.append(i)
        text_list.append(text)

    # id_list = id_list[:500]
    # text_list = text_list[:500]

    g = tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        embed = hub.Module("https://tfhub.dev/google/elmo/2")
        my_result = embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

    # Create session and initialize.
    # config = tf.ConfigProto(
    #     device_count={'GPU': 0}
    # )
    # session = tf.Session(graph=g, config=config)
    session = tf.Session(graph=g)
    session.run(init_op)
    # Loop over batches
    vector_result =[]
    for batch in get_batches_examples_for_elmo(text_list, batch_size=batch_size):
        # Feed dictionary
        # feed = {text_input: x}
        re_x = session.run(my_result, feed_dict={text_input: batch})
        vector_result.extend(re_x)
    assert len(id_list) == len(vector_result)
    session.close()

    return dict(zip(id_list, vector_result))

def get_multi_embeddings_from_elmo(examples, batch_size=100):
    # Create graph and finalize (optional but recommended).
    concept_dict = {}
    for example in examples:
        if example.id_a not in concept_dict:
            concept_dict[example.id_a] = example.text_a
        if example.id_b not in concept_dict:
            concept_dict[example.id_b] = example.text_b

    id_list = []
    text_list = []
    for i, text in concept_dict.items():
        id_list.append(i)
        text_list.append(text)

    # id_list = id_list[:500]
    # text_list = text_list[:500]
    # print(len(text_list))
    # print(text_list[1090:1095])
    print("get embeddings from the first source: elmo")

    g = tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        embed = hub.Module("https://tfhub.dev/google/elmo/2")
        my_result = embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

    # Create session and initialize.
    # config = tf.ConfigProto(
    #     device_count={'GPU': 0}
    # )
    # session = tf.Session(graph=g, config=config)
    session = tf.Session(graph=g)
    session.run(init_op)
    # Loop over batches
    vector_result =[]

    for batch in get_batches_examples_for_elmo(text_list, batch_size=batch_size):
        # Feed dictionary
        re_x = session.run(my_result, feed_dict={text_input: batch})
        vector_result.extend(re_x)
    assert len(id_list) == len(vector_result)
    session.close()

    print("get embeddings from the second source: universal-sentence-encoder-large v3")

    # tf.reset_default_graph()
    g2 = tf.Graph()
    with g2.as_default():
        text_input_2 = tf.placeholder(dtype=tf.string, shape=[None])
        embed2 = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        my_result_2 = embed2(text_input_2)
        init_op_2 = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g2.finalize()

    session_2 = tf.Session(graph=g2)
    session_2.run(init_op_2)
    vector_result_2 = []
    count = 0
    for batch in get_batches_examples_for_elmo(text_list, batch_size=batch_size):
        # Feed dictionary
        count += 1
        # print("working on batch {}".format(count))
        re_x_2 = session_2.run(my_result_2, feed_dict={text_input_2: batch})
        vector_result_2.extend(re_x_2)
    assert len(id_list) == len(vector_result_2)

    session_2.close()
    # print(len(vector_result))
    # print(vector_result[1090:1095])
    return dict(zip(id_list, vector_result)), dict(zip(id_list, vector_result_2))


def get_batches_examples_for_elmo(examples, batch_size=500, random_flag=False):
    samples = examples
    num_samples = len(samples)
    if random_flag:
        shuffle(samples)
    for offset in range(0, num_samples, batch_size):
        batch_samples = samples[offset:offset + batch_size]
        yield batch_samples


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


def column(matrix, i):
    return [row[i] for row in matrix]


