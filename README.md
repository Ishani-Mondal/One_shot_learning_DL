# One_shot_learning_DL
Deep Learning Term Project
Team 9

Contributors:
1. Ishani Mondal (17CS71P01)
2. Sinchani Chakraborty(17CS71P02)
3. Pranesh Santikellur(17CS91P06)
4. Paheli Bhattacharya(17CS92R05)

Details about the source files:
1. Omniglot_Data : Contains the dataset of 50 alphabets divided into background and evaluation set
2. data_loader.py : This file loads the dataset from a specified path and creates batches for the purpose of training and one-shot testing
3. image_augmentor.py : Augments the images with various affine transformations
4. siamese.py: Actual model of the Convolutional Neural Network
5. main.py: Trains the model by loading the data into siamese network and performs the one-shot testing
6. create_dataset_for_knn.py : modifies the dataset for implementing knn on the data. The code creates a new folder "knn_new" in the working diectory and merges all the images from images_background and images_evaluation folders of the Omniglot Dataset with a change in the image filenames <label.imagename.png> where labels = {0,...,49} 
7. knn.py : the code runs K-Nearest Neighbour classifier on the data present in folder "knn_new". It takes raw image pixels as input features. As arguments the dataset path and no. of neighbours (optional, default=1) are expected

Pre-requisites:
1. Keras
2. Tensorflow
3. PIL
4. OpenCV

Instructions to run the KNN Model:
---1. python create_dataset_for_knn.py
---2. python knn.py <dataset_path>

Instructions to train and test the siamese Network Model:
----python main.py 











