import tensorflow as tf
import numpy as np
import ClassifyBaseModel

class FullyConnectedNet(ClassifyBaseModel):
    """
        This employs a Softmax Classifier for multilabel classification
        We will employ a simple Softmax Model to classify the captions
    """

    def __init__(self, train_features, train_labels, validation_features, validation_labels,config):
        """

        Args:
            train_features: numpy ndarray of shape (N, D)
            train_labels: numpy ndarray of shape (N)
            N - Number of training examples
            D - Dimension of the data
            config: dictionary containing the hyperparameters
                    batch_size
                    learning_rate
                    learn_type
                    log_folder - Folder into which the logs need to be written
                    num_epochs
                    num_hidden_units - The number of neurons in hidden layers
                    reg - regularization strength
                    test_log_folder   ----
                                          |--->  This is for storing logs  that needs to be separated into
                                          |--->  test and train. For eg. Training versus validation accuracies
                    train_log_folder  ----
        """
        super(FullyConnectedNet, self).__init__(train_features, train_labels,
                                                validation_features, validation_labels,
                                                config)
        self.reg = config["reg"]
        self.num_hidden_units = config["num_hidden_units"]
        self.num_layers = config["num_layers"]
        self.learning_rate = config["learning_rate"]
        self.learn_type = config["learn_type"]
        self.log_folder = config["log_folder"]
        self.train_log_folder = config["train_log_folder"]
        self.test_log_folder = config["test_log_folder"]
        self.num_epochs = config["num_epochs"]
        self.batch_size = config["batch_size"]
        # Form the graph for execution
        # 1. Add the placeholders that represent the end points of the graph
        # 2. Initialise the parameters of the model
        # 3. Calculate the scores - The normalised probabilities of a data belonging to C classes
        # 4. Add the loss operation - The cross entropy loss that is needed
        # 5. Add the accuracy operation
        # 6. ADD THE SUMMARIES OPERATION
        # 7. Add the summary writers

        # Over riding from the base class
        # It is specific to this class
        self.num_train, self.num_dimensions = train_features.shape

        self.add_placeholder()
        self.initialize_parameters()
        self.yhat = self.calculate_scores()
        self.loss, self.loss_summary = self.add_loss()
        self.train_op = self.add_optimiser(type=self.learn_type)
        self.calculate_accuracy_op, self.accuracy_summary = self.calculate_accuracy()
        self.merged = self.add_summaries_operation()
        self.summary_writer = tf.train.SummaryWriter(self.log_folder)
        self.train_summary_writer = tf.train.SummaryWriter(self.train_log_folder)
        self.test_summary_writer = tf.train.SummaryWriter(self.test_log_folder)

    def add_placeholder(self):
        """

        Returns:

        """
        with tf.name_scope("Inputs") as scope:
            self.X  = self.get_placeholder([None, self.num_dimensions])
            self.y = self.get_placeholder([None, self.num_classes])
            self.keep_prob = tf.placeholder(dtype=tf.float32)

    def add_loss(self):
        """
        THE TRUE DISTRIBUTION WILL BE A PLACEHOLDER
        Args:
            yhat: prediction of shape (N, C)
                  N - number of training examples
                  C - Number of classes

        Returns: loss - scalar

        """
        # IN WHAT CASES WOULD tf.log() REACH A 0 VALUE
        # IF IT REACHES 0, IT WILL BE GIVE A NAN
        with tf.name_scope("loss") as scope:
            loss = 0.0
            loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.yhat), reduction_indices=[1]))
            for i in xrange(1, self.num_layers):
                with tf.variable_scope("layer_%s"%i, reuse=True):
                  loss = loss + self.reg * tf.nn.l2_loss(tf.get_variable("W"))

            loss_summary = tf.scalar_summary("loss_summary", loss)
            return loss, loss_summary

    def add_optimiser(self, type="vanilla"):
        """

        Add the optimizer function to perform Gradient Descent
        Args:
            type: The type of update that is needed
                  ["vanilla", "adam", "adagrad", "rmsprop"]
        Returns: None
        """
        if type not in ["vanilla", "adam", "adagrad", "rmsprop"]:
            raise ValueError("Please provide any of [vanilla, adam, adagrad, rmsprop] for optimisation")

        with tf.name_scope("gradient_descent") as scope:
            train_op =  None
            if type == "vanilla":
                train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            elif type == "adam":
                train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            elif type == "adagrad":
                train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
            elif type == "rmsprop":
                train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            return train_op

    def add_summaries_operation(self):
        return tf.merge_summary([self.loss_summary, self.accuracy_summary])

    def calculate_accuracy(self):

        with tf.name_scope('accuracy') as scope:
            correct_prediction = tf.equal(tf.argmax(self.yhat, 1), tf.argmax(self.y, 1))
            accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
            accuracy_summary = tf.scalar_summary("accuracy", accuracy)
            return accuracy, accuracy_summary

    def calculate_scores(self):
        """
        Return the scores of the model
        If it is convolutional neural networks or neural networks
        calculate the activations and final scores before the softmax loss is
        added
        """
        out = self.X

        for i in xrange(1, self.num_layers):
            with tf.variable_scope("layer_%s"%i, reuse=True) as scope:
                weights = tf.get_variable("W")
                biases = tf.get_variable("b")
###  ADDED DROPOUT TO THE FOLLOWING LINE 
                out = tf.nn.dropout(tf.nn.relu(tf.matmul(out, weights) + biases), keep_prob)
        with tf.variable_scope("layer_%s"%(self.num_layers), reuse=True) as scope:
            weights = tf.get_variable("W")
            biases = tf.get_variable("b")
            scores = tf.nn.softmax(tf.matmul(out, weights) + biases)

        return scores

    def initialize_parameters(self):
        """
        Initialize the parameters of the model
        Returns:

        """
        indices_list = []
        indices_list.append(self.num_dimensions)
        indices_list.extend([self.num_hidden_units] * (self.num_layers-1))
        indices_list.append(self.num_classes)

        for i in xrange(1,self.num_layers+1):
            with tf.variable_scope("layer_%s"%i) as scope :
                W = tf.get_variable("W", shape=[indices_list[i-1], indices_list[i]],
                                         initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b", shape=indices_list[i], initializer=tf.constant_initializer(0.0))

    def test(self, test_features, test_labels):
        num_test = test_features.shape[0]
        one_hot_labels = np.zeros([num_test, self.num_classes])
        one_hot_labels[range(num_test), test_labels] = 1
        return self.session.run(self.calculate_accuracy_op, feed_dict={self.X:test_features, self.y:one_hot_labels,
                                                             self.keep_prob:0.5})

    def train(self,print_every=1, log_summary=True):
        """

        Args:
            session: The tensor flow session on which the various operations can be run
            in parallel
            print_every: Print the loss and the epoch number every 10 iterations

        Returns:
            pass

        """
        init_operation = tf.initialize_all_variables()
        self.session.run(init_operation)
        num_iterations = 0
        best_validation_accuracy = -np.inf
        best_weights = None
        best_bias = None
        total_partitions = int(np.ceil(float(self.num_train) / self.batch_size))

        # 1. For every epoch
        #     Pass once through all the training examples
        # 2. At the end of every epoch get the validation and train accuracy
        for i in xrange(self.num_epochs):
            for iteration in range(total_partitions):
                train_data, train_labels = self.get_random_batch(self.batch_size)
                if i == 0 and num_iterations == 0:
                    print "*"*80
                    loss = self.session.run(self.loss,
                                            feed_dict={self.X: train_data,
                                                       self.y: train_labels,
                                                       self.keep_prob:0.5})
                    print "initial loss %f" % (loss,)
                    print "*" * 80
                _, loss_summary,loss  = self.session.run([self.train_op, self.merged, self.loss],
                                           feed_dict={self.X: train_data,
                                                      self.y: train_labels,
                                                      self.keep_prob: 0.5})
                if num_iterations % print_every == 0 and log_summary == True:
                    print "loss at iteration %d is %f" % (num_iterations, loss)
                num_iterations += 1
                self.summary_writer.add_summary(loss_summary, num_iterations)

            # AT THE END OF EVERY EPOCH CALCULATE AND PRINT THE TRAINING AND THE VALIDATION ACCURACY
            training_summary, training_accuracy = self.session.run([self.merged, self.calculate_accuracy_op],
                                                   feed_dict={self.X: self.train_data,
                                                              self.y: self.train_labels,
                                                              self.keep_prob: 0.5})

            validation_summary, validation_accuracy = self.session.run([self.merged, self.calculate_accuracy_op],
                                                    feed_dict={self.X: self.validation_data,
                                                               self.y: self.validation_labels,
                                                               self.keep_prob: 0.5})

            print "*" * 80
            print "Training accuracy at the end of epoch %i: %f" % (i, training_accuracy)
            print "Validation accuracy at the end of epoch %i %f" % (i, validation_accuracy)
            print "*" * 80

            self.train_summary_writer.add_summary(training_summary, i)
            self.test_summary_writer.add_summary(validation_summary, i)

            # TODO: STORE THE BEST PARAMETERS

        return validation_accuracy
