# Emiliano Portfolio
Machine Learning and Data science projects


# [Project 1: Tennis Match Analysis](https://github.com/emipizana/projectCV/tree/FinalBranch)

## 📁 Project Structure

```
tennis_analysis/
├── src/
│   └── tennis_analysis/
│       ├── downloader/      # Video downloading
│       ├── preprocessor/    # Point segmentation
│       ├── tracking/        # Ball and player tracking
│       ├── projector/       # Court Projector of minimap
│       └── postprocessing/  # Visualization
├── models/                  # Pre-trained models
├── examples/               # Usage examples
└── tests/                 # Test suite
```
## 🔧 Pipeline Steps

1. **Download**: Downloads the tennis match video, optionally selecting a specific segment
2. **Preprocessing**: 
   - Segments the video into individual points
   - Detects scene changes
   - Validates point durations and characteristics
3. **Tracking**:
   - Tracks the ball using YOLO with trajectory prediction
   - Tracks and identifies players
   - Handles occlusions and missed detections
4. **Projector**:
   - Edge detection for the dimension of the tennis courts.
   - Uses the players positions to project them in mini-map.
   - Generates a mini-map in the video.
5. **Postprocessing**:
   - Generates visualization overlays
   - Creates analysis summaries
   - Exports processed videos


## 📊 Output

The framework generates:
1. **Points Directory**: Individual video clips for each tennis point
2. **Post-processed Points**: Processed videos with ball, player tracking and mini-map visualization
3. **Analysis Data**: JSON files containing tracking data and statistics
4. **Visualizations**: Optional plotting of trajectories and statistics

![](/images/Tennis_Example_picture.png)


# [Project 2: Nueral Network from Scratch](https://github.com/emipizana/Neural-Network-from-Scratch)

## Objective

The objective of the project is to build a Neural Network only using NumPy and Pandas. Pandas is mainly used for reading data, in this case we are using the MNIST dataset. The main point of the project was to focus on the theory of the Neural Network more than handling a diffucult dataset and hyperparameter tunning. Finally, we also imported matplotlib for plotting the images and Keras to import the MNIST dataset. The project is divided in two, the first part is the Neural Network from scratch, while the second part is building a neural network using keras with the same architecture as the one before. Finally, we compare both results to see how well is the "manual" neural network works. 

## MNIST Data Set

We have our data set divided by X_train and X_test with shapes (60000, 28, 28) and (10000, 28, 28) respectively, in which each observation is a image with 28x28 pixels. The target data divided by Y_train and Y_test with shapes of (60000,) and (10000,) respetively. The next image is an example from the train data set. 

![](/images/Correct_classification.png)

## Architecture of the Nueral Network.
![](/images/NN_Architecture.jpg)

## Accuracy and training on Train Data set
![](/images/Train_accuracy.JPG)

For this Neural Network we manage to get:

![](/images/Accuracy_test_data.JPG)

## Misclassify images:
![](/images/Miss_Classify_1.JPG)
![](/images/Miss_Classify_2.JPG)
![](/images/Miss_Classify_3.JPG)

We can see above three examples of misclassify images, most of them are not that clear, and can even be misclassified by humans.

## Confusion Matrix
This was the only part that we used sklearn

![](/images/Confusion_matrix_NN_1.JPG)


## Neural Network with Keras

For the Neural Network using keras we used the same architecture but the training was different, we used:
* Optimizer: Adam
* Loss function: Categorical Crossentropy
* Batch Size: 32

![](/images/keras_model.JPG)

## Training and accuracy

![](/images/5_epochs_keras_train.JPG)
![](/images/Accuracy_graph_keras.JPG)

We manage an accuracy of:

![](/images/accuracy_keras_test.JPG)

## Conclusion

To begin with, when constructing a neural network from the ground up, there are various factors to take into account in order to make the model more robust. For example, we can modify the loss function to categorical crossentropy, opt for a more efficient optimization method beyond gradient descent, utilize batches instead of passing through the entire dataset, and cautiously incorporate additional hidden layers without overfitting.

The primary emphasis of the project was to gain a deeply understanding of neural network concepts, particularly the backward function. Despite this focus, we achieved a robust model with an accuracy exceeding 91%. While the Keras model had a higher accuracy, surpassing 97%, our model still performed admirably with a notably high accuracy level.

##Reference:

Aflak, O. (2018, November 14). Neural Network from scratch in Python. Towards Data Science. https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

# [Project 3: Image-Classification](https://github.com/emipizana/Image_Classification/tree/main)

## Objective

The point of the project is to classify images. First we have several images like the ones below.

![](/images/Ejemplo_1.jpg)
![](/images/Ejemplo_2.jpg)

We have seven types of bills €5, €10, €20, €50, €100, €200, and €500. The data is divided in train and test data, the goal is to classify the bills as well as we can.
## Method used

* Python
* Keras
* Convolutional Neural Network

## Approach

Most of the time you need several layers to build a strong neural network, so for this project we used two different Convolutional Neural Networks already builded in the Keras library. The first one is the ResNet50 and the second one is the VGG16, after importing them we changed the output layer to have an output of the seven classes we need.

## ResNet50

For the ResNet50 we had:<br>
* Total params:23,602,055<br>
* Trainable params: 14,343<br>
* Non-trainable params: 23,587.712 <br>
* Accuracy per Epoch: 
  
![](/images/ACC_RESNET.jpg)

Loss per Epoch:<br>
![](/images/LOSS_RESNET.jpg)


## VGG16

For the VGG16 we had:<br>
* Total params:134.289.223<br>
* Trainable params: 28,679<br>
* Non-trainable params: 134,260,544<br>
* Accuracy per Epoch:<br>
![](/images/ACC_VGG.jpg)

Loss per Epoch:<br>
![](/images/LOSS_VGG.jpg)


We can see in the next image how the bills are classified.

![](/images/PREDICTION_VGG.jpg)


## Conclusion

Finally using the test data, the accuracy of both models were:<br>
* ResNet50: .5714<br>
* VGG16: .9857<br>
We can conclude that the Neural Network VGG16 was much better!!

## Reference
IBM DL0320EN Applied Deep Learning Capstone Project


# [Project 4: Tweet Classification](https://github.com/emipizana/Project-Tweet-Classification)

## Objective

Twitter has become an important communication channel in times of emergency. We have several tweets and the point is predicting whether a given tweet is about a real disaster or not. We will classify disaster tweets as 1, and non-disaster tweets as 0.
Here are some examples of the tweets we have to classify,

![](/images/Tweet_examples.jpeg)

Here we have the first 5 rows of our training data frame,
![](/images/tweet_images.JPG)

## Method used

* Python
* Keras
* RNN
* BERT Embedding for NLP

## Approach

For this project we builded one simple Recurrent neural network to classify tweets and then we compared the results against the BERT model uploaded from keras. We can see how the BERT model performs much better but it is also computationally more expensive.

## First RNN

we managed an accuracy of 73.93% of accuracy over 20 epochs. Now we can see the accuracy and loss per epoch,

![](/images/Net_1_acc.JPG)
![](/images/Net_1_loss.JPG)

It is clear that this RNN is not a good model for this data, and there are a lot of things to improve in how to build a stronger RNN.

##BERT model

For the BERT model we managed a 83.06% of accuracy over 4 epochs. We can see the accuracy and loss per epoch,

![](/images/Net_2_acc.JPG)
![](/images/Net_2_loss.JPG)

This model was computational expensive so we tried it with just 4 epoches. It was for sure a better fit to classify the tweets, however, there are several things we can improve, especially the hyperparameters. 



# [Project 5: Restaurant predicting sales](https://github.com/emipizana/Project-1)
The purpose of this project is to predict the revenue of a restaurant around the world. In this case the prediction will depend on the restaurants open date, location, city type and several other variables.

## Methods Used
* Data Visualization
* Random Forest
* Multiple Regression

![](/images/Rplot02.png)

# [Project 6: Predicting Titanic survivors](https://github.com/emipizana/Project-2)
The purpose of this project is to predict if a passenger survived the sinking of the Titanic or not. In this case the prediction will be a binary variable that indicates if the passenger survived. 

## Methods Used
* Data Visualization
* Machine Learning
* Logistic Regression

![](/images/Rplot05.png)


# [Project 7: Population Prediction](https://github.com/emipizana/Project_3)
In this project we read some data from https://www.fao.org/faostat/en/#data, representing the urban, rural, male, female and total population from several conuntries around the world. After data visualization and a summary we start by using a time serie to predict china´s urban population and compare it to a linear regression.

![](/images/China_population.png)

## Methods Used
* Data Visualization
* Time series
* Linear Regression

![](/images/TS_China_pred.png)


# [Project 7: MNIST Prediction](https://github.com/emipizana/Project_4)
In this project we take the MNIST database of handwritten digits, it has a training set of 60,000 examples, and a test set of 10,000. The images can be seen as:

![](/images/Four_image)

The objective of the project is to be able to predict the numbers that are handwritten in the test data. Our first two models are binary classifiers, SGDClassifier and Random Forest build to predict if each image is a 4. 
## Methods Used
* SGDCLassifier
* Random Forest
* Cross Validation 
* Metrics (Precision and recall score, ROC score)

After that we use KNN Classifier (multiple classifier) to predict the number of each image.

## F1 score using different K values (k as the n_neighbors). 
![](/images/KNN_Classifier.png)

## Confusion Matrix for the KNN Classifier
![](/images/Conf_matrix)
* More than 97% accuracy with k = 3. 

## Methods Used
* KNN Classifier
* Cross Validation 
* Metrics (Precision and recall score, ROC score, F1 score)

## Project based on
Géron, A. (2022, November 15). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (3rd ed.). O’Reilly Media.

