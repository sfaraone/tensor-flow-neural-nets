Fully Connected Neural Nets using Tensorflow 
------------------------------------------------------------------------------------------------
Files 
------------------------------------------------------------------------------------------------
ClassifyBaseModel.py - Used as a base class for different kind of neural networks 

FullyConnectedNet.py - Fully Connected n layer neural network can be constructed using this 

------------------------------------------------------------------------------------------------
INITIALISATION 
------------------------------------------------------------------------------------------------
train_features of shape (N,[a,b,c])


train_labels os shape (N,)


validation_features - same shape as train features 


validation_labels - same shape as train labels 


CONFIG - contains the hyper parameter setup 

- batch_size
- learning_rate
- learn_type - either of vanilla, adam, adagrad or rmsprop
- num_epochs - Number of epochs to train 
- num_hidden_units - NUmber of hidden units in the hidden layers 
- num_layers - NUmber of  layers in the network 
- log_folder - Folder into which the logs need to be written
- reg - refularization strength
- test_log_folder   ----
                        |--->  This is for storing logs  that needs to be separated into
                        |--->  test and train. For eg. Training versus validation accuracies
- train_log_folder  ----
