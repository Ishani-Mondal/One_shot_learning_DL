import os

'''Import keras based modules '''
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
import keras.backend as K

'''Modules required for debugging and analysis'''
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Learning Rate multipliers for each layer
learning_rate_multipliers = {}
learning_rate_multipliers['Conv1'] = 1
learning_rate_multipliers['Conv2'] = 1
learning_rate_multipliers['Conv3'] = 1
learning_rate_multipliers['Conv4'] = 1
learning_rate_multipliers['Dense1'] = 1


input_shape = (105, 105, 1)  # Size of images
model = []


def model_architecture():
        global model
        learning_rate = 10e-4
        # The Sieamese architecture
        siamese_network = Sequential()
        siamese_network.add(Conv2D(filters=64, kernel_size=(10, 10),
                                     activation='relu',
                                     input_shape=input_shape,
                                     kernel_regularizer=l2(0.01),
                                     name='Conv1'))
        siamese_network.add(MaxPool2D())

        siamese_network.add(Conv2D(filters=128, kernel_size=(7, 7),
                                     activation='relu',
                                     kernel_regularizer=l2(0.01),
                                     name='Conv2'))
        siamese_network.add(MaxPool2D())

        siamese_network.add(Conv2D(filters=128, kernel_size=(4, 4),
                                     activation='relu',
                                     kernel_regularizer=l2(0.01),
                                     name='Conv3'))
        siamese_network.add(MaxPool2D())

        siamese_network.add(Conv2D(filters=256, kernel_size=(4, 4),
                                     activation='relu',
                                     kernel_regularizer=l2(0.01),
                                     name='Conv4'))
        siamese_network.add(MaxPool2D())

        siamese_network.add(Flatten())
        siamese_network.add(
            Dense(units=4096, activation='sigmoid',
                  kernel_regularizer=l2(0.0001),
                  name='Dense1'))

        # Now imput  pairs of images
        input_image_1 = Input(input_shape)
        input_image_2 = Input(input_shape)

        processed_image_1 = siamese_network(input_image_1)
        processed_image_2 = siamese_network(input_image_2)

        # L1 distance layer between the two encoded outputs
        l1_distance_layer = Lambda(
            lambda tensors: K.abs(tensors[0] - tensors[1]))
        abs_diff = l1_distance_layer([processed_image_1, processed_image_2])

        # Same class or not prediction
        prediction = Dense(units=1, activation='sigmoid')(abs_diff)
        model = Model(
            inputs=[input_image_1, input_image_2], outputs=prediction)

        # Define the optimizer and compile the model
        '''        
        optimizer = Modified_SGD(
            lr=learning_rate,
            lr_multipliers=learning_rate_multipliers,
            momentum=0.5)
        '''
        optimizer = SGD(
            lr=learning_rate,
            momentum=0.5)
        model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],
                           optimizer=optimizer)




def train_model(omniglot_loader,number_of_iterations, support_set_size,
                              final_momentum, momentum_slope, evaluate_each,
                              model_name):


        omniglot_loader.split_train_datasets()

        # Variables that will store 100 iterations losses and accuracies
        # after evaluate_each iterations these will be passed to tensorboard logs
        train_losses = np.zeros(shape=(evaluate_each))
        train_accuracies = np.zeros(shape=(evaluate_each))
        count = 0
        earrly_stop = 0
        # Stop criteria variables
        best_validation_accuracy = 0.0
        best_accuracy_iteration = 0
        validation_accuracy = 0.0
        #self.model.save_weights('models/' + model_name + '.h5')
        #print('Model Name %s:',model_name)
        # Train loop
        for iteration in range(number_of_iterations):

            # train set
            images, labels = omniglot_loader.get_train_batch()
            
            train_loss, train_accuracy = model.train_on_batch(
                images, labels)

            # Decay learning rate 1 % per 500 iterations (in the paper the decay is
            # 1% per epoch). Also update linearly the momentum (starting from 0.5 to 1)
            if (iteration + 1) % 500 == 0:
                K.set_value(model.optimizer.lr, K.get_value(
                    model.optimizer.lr) * 0.99)
            if K.get_value(model.optimizer.momentum) < final_momentum:
                K.set_value(model.optimizer.momentum, K.get_value(
                    model.optimizer.momentum) + momentum_slope)

            
            train_losses[count] = train_loss
            train_accuracies[count] = train_accuracy
            #print(train_losses)
            # validation set
            count += 1
            print('Iteration %d/%d: Train loss: %f, Train Accuracy: %f, lr = %f' %
                  (iteration + 1, number_of_iterations, train_loss, train_accuracy, K.get_value(
                      model.optimizer.lr)))

            # Each 100 iterations perform a one_shot_task and write to tensorboard the
            # stored losses and accuracies
            if (iteration + 1) % evaluate_each == 0:
                number_of_runs_per_alphabet = 40
                # use a support set size equal to the number of character in the alphabet
                validation_accuracy = omniglot_loader.one_shot_test(
                    model, support_set_size, number_of_runs_per_alphabet, is_validation=True)
                '''
                self.__write_logs_to_tensorboard(
                    iteration, train_losses, train_accuracies,
                    validation_accuracy, evaluate_each)
                '''
                count = 0
                
                # Some hyperparameters lead to 100%, although the output is almost the same in 
                # all images. 
                if (validation_accuracy == 1.0 and train_accuracy == 0.5):
                    print('Early Stopping: Gradient Explosion')
                    print('Validation Accuracy = ' +
                          str(best_validation_accuracy))
                    return 0
                elif train_accuracy == 0.0:
                    return 0
                else:
                    # Save the model
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        best_accuracy_iteration = iteration
                        
                        model_json = model.to_json()

                        if not os.path.exists('./models'):
                            os.makedirs('./models')
                        with open('models/' + model_name + '.json', "w") as json_file:
                            json_file.write(model_json)
                        model.save_weights('models/' + model_name + '.h5')

            # If accuracy does not improve for 10000 batches stop the training
            if iteration - best_accuracy_iteration > 10000:
                print(
                    'Early Stopping: validation accuracy did not increase for 10000 iterations')
                print('Best Validation Accuracy = ' +
                      str(best_validation_accuracy))
                print('Validation Accuracy = ' + str(best_validation_accuracy))
                break

        print('Training Ended!')
        return best_validation_accuracy



