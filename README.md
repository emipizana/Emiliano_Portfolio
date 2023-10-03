# Emiliano_Portfolio
Data science projects

# [Project-Image_Classification](https://github.com/emipizana/Image_Classification/tree/main)

## Objective

The point of the project is to classify images. First we have several images like the ones below.

![](/images/Ejemplo_1.jpg)
![](/images/Ejemplo_2.jpg)

We have seven types of bills €5, €10, €20, €50, €100, €200, and €500. The data is divided in train and test data, the goal is to classify the bills as well as we can.

## Approach

Most of the time you need several layers to build a strong neural network, so for this project we used two different Convolutional Neural Networks already builded in the Keras library. The first one is the ResNet50 and the second one is the VGG16, after importing them we changed the output layer to have an output of the seven classes we need.

## ResNet50

<p> For the ResNet50 we had:<br>
<p> Total params:23,602,055<br>
<p> Trainable params: 14,343<br>
<p> Non-trainable params: 23,587.712 <br>
<p> Accuracy per Epoch: <\p> 
  
![](/images/ACC_RESNET.jpg)

Loss per Epoch:<br>
![](/images/LOSS_RESNET.jpg)


## VGG16

For the VGG16 we had:
Total params:134.289.223
Trainable params: 28,679
Non-trainable params: 134,260,544
Accuracy per Epoch:
![](/images/ACC_VGG16.jpg)

Loss per Epoch:
![](/images/LOSS_VGG16.jpg)


We can see in the next image how the bills are classified.
![](/images/PREDICTION_VGG.jpg)


## Conclusion

Finally using the test data, the accuracy of both models were:
ResNet50: .5714
VGG16: .9857
We can conclude that the Neural Network VGG16 was much better!!



# [Project-1](https://github.com/emipizana/Project-1)
The purpose of this project is to predict the revenue of a restaurant around the world. In this case the prediction will depend on the restaurants open date, location, city type and several other variables.

## Methods Used
* Data Visualization
* Random Forest
* Multiple Regression

![](/images/Rplot02.png)

# [Project-2](https://github.com/emipizana/Project-2)
The purpose of this project is to predict if a passenger survived the sinking of the Titanic or not. In this case the prediction will be a binary variable that indicates if the passenger survived. 

## Methods Used
* Data Visualization
* Machine Learning
* Logistic Regression

![](/images/Rplot05.png)


# [Project-3](https://github.com/emipizana/Project_3)
In this project we read some data from https://www.fao.org/faostat/en/#data, representing the urban, rural, male, female and total population from several conuntries around the world. After data visualization and a summary we start by using a time serie to predict china´s urban population and compare it to a linear regression.

![](/images/China_population.png)

## Methods Used
* Data Visualization
* Time series
* Linear Regression

![](/images/TS_China_pred.png)


# [Project-4](https://github.com/emipizana/Project_4)
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

