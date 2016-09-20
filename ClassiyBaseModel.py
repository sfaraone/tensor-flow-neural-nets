import numpy as np
import tensorflow as tf
import time
import random


class ClassifyBaseModel(object):
    """
        This models the base class that is required for
        all the classifications in Tensor flow

        Tensorflow classifications model
        Inputs as placeholders
        Output labels as placeholder
    """

    def __init__(self, training_features, train_labels,
                 validation_features, validation_labels,
                 config):
        print "Initialising the session in the base classify model"
        self.train_data = training_features
        self.validation_data = validation_features
        self.config = config
        C = np.unique(train_labels).shape[0]
        self.num_classes = C
        self.num_train = train_labels.shape[0]
        self.train_labels = self.make_it_hot(train_labels)
        self.validation_labels = self.make_it_hot(validation_labels)
        self.session = tf.Session()

        print "Number of training examples %d" % (self.num_train)

    def add_loss(self):
        """
        THE TRUE DISTRIBUTION WILL BE A PLACEHOLDER
        Returns: loss - scalar

        """
        pass

    def add_summaries_operation(self):
        pass

    def add_placeholder(self):
        """
        Add all the placeholder that are required for the model here
        Returns:
        """
        pass

    def calculate_accuracy(self):
        """
        Add the operation to calculate the accuracy for your model
        Returns:

        """
        pass

    def calculate_scores(self):
        """
        Return the scores of the model
        If it is convolutional neural networks or neural networks
        calculate the activations and final scores before the softmax loss is
        added
        """
        pass

    def get_batch(self, batch_size):
        """
        Args:
            batch_size: an integer
        Returns: The next batch of training data and training labels
        and all that is needed for training the minibatch

        """
        # 1. Shuffle the indices
        # 2. Randomly pick batch_size number of integers from the shuffled set of indices
        # 3. Get the corresponding data
        # 4. Convert the ys into a one hot vector for calculating the loss
        indices = np.arange(self.num_train)
        np.random.shuffle(indices)
        num_partitions = int(np.ceil(float(self.num_train) / batch_size))
        partitions = self.partition_lists(indices.tolist(), num_partitions)
        processed_training_examples = 0

        for partition in partitions:
            processed_training_examples += len(partition)
            train_data = self.train_data[partition]
            train_labels = self.train_labels[partition]
            assert np.isnan(np.sum(train_data)) == False
            assert np.isnan(np.sum(train_labels)) == False
            yield train_data, train_labels
        assert processed_training_examples == self.num_train

    def get_random_batch(self, batch_size):

        random.seed(time.time())
        indices = np.arange(self.num_train)
        np.random.shuffle(indices)
        num_partitions = int(np.ceil(float(self.num_train) / batch_size))
        partitions = self.partition_lists(indices.tolist(), num_partitions)

        random_partition_index = np.random.randint(0, high=len(partitions), size=1)
        random_partition = partitions[random_partition_index]
        train_data = self.train_data[random_partition]
        train_labels = self.train_labels[random_partition]
        assert np.isnan(np.sum(train_data)) == False
        assert np.isnan(np.sum(train_labels)) == False
        return train_data, train_labels

    def read_csv(self, batch_size, filename, record_defaults):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue) 
        decoded = tf.decode_csv(value, record_defaults = record_defaults)
        return tf.train.shuffle_batch(decoded,
                                      batch_size=batch_size,
                                      capacity=batch_size * 50,
                                      min_after_dequeue=batch_size)

    def features_labels(self, batch_size, filename, record_defaults):
        sepal_length, sepal_width, petal_length, petal_width, flower = \
            read_csv(batch_size, filename, record_defaults) 
        features = tf.transpose(tf.pack([sepal_length, sepal_width, petal_length,petal_width]))
        labels = tf.equal(flower, ["Iris-setosa"])
        labels = tf.reshape(labels, [batch_size, 1])
        return features, labels
    
    def get_placeholder(self, size, dtype=tf.float32):
        """
        SHOULD NOT BE CHANGED IN THE INHERITED CLASS
        Args:
            size: The size of the placeholder to be created
            dtype: Data type for the placeholder

        Returns: tf.placeholder
        """
        return tf.placeholder(dtype, size)

    def initialize_parameters(self):
        """
        Initialize the parameters of the model
        Returns:

        """
        pass

    def make_it_hot(self, labels):
        num_examples = labels.shape[0]
        one_hot_labels = np.zeros((num_examples, self.num_classes), dtype=np.float32)
        one_hot_labels[range(num_examples), labels] = 1.0
        return one_hot_labels

    def test(self, test_features, test_labels):
        """
        Calculates the acuracy on the test data set that is passed
        Args:
            test_features: Test features of shape (N, D)
            test_labels: : Test labels of shape (N)

        Returns: test_accuracy

        """
        pass

    def train(self):
        """

        Args:
            session: The tensor flow session on which the various operations can be run
            in parallel

        Returns:
            pass

        """

    def close_session(self):
        self.session.close()

    def partition_lists(self, lst, n):
    """
        given a list it is partitioned into n sub lists
    """
    division = len(lst) / float(n)
    return [lst[ int(round(division * i)) : int(round(division * (i+1))) ] for i in xrange(n)]
