"""
Team 10
One-Shot Learning

Used to train the network

"""
from data_loader import OmniglotLoader
from siamese import model_architecture
from siamese import train_model


def main():
    path_of_dataset = 'Omniglot Dataset'
    augmentation_to_be_used = True
    learning_rate = 10e-4
    batch_size = 32 
    
    ''' Load the Dataset '''
    omniglot_loader = OmniglotLoader(path_of_dataset=path_of_dataset, augmentation_to_be_used=augmentation_to_be_used, batch_size=batch_size)
    ''' Construct the architecture '''
    model = model_architecture()
    
    '''   Hyper Parameters '''
    momentum = 0.9
    # linear epoch slope evolution
    momentum_slope = 0.01
    support_set_size = 20
    evaluate_each = 1000
    number_of_train_iterations = 1000000

    '''   Train the Siamese '''
    validation_accuracy = train_model(omniglot_loader,number_of_iterations=number_of_train_iterations,
                                                                support_set_size=support_set_size,
                                                                final_momentum=momentum,
                                                                momentum_slope=momentum_slope,
                                                                evaluate_each=evaluate_each, 
                                                                model_name='team10')
    if validation_accuracy == 0:
        evaluation_accuracy = 0
    else:
        # Load the weights with best validation accuracy
        siamese_network.model.load_weights('.models/team10.h5')
        evaluation_accuracy = siamese_network.omniglot_loader.one_shot_test(siamese_network.model,
                                                                        20, 40, False)
    
    print('Final Evaluation Accuracy = ' + str(evaluation_accuracy))


if __name__ == "__main__":
    main()
