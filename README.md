# **Traffic Sign Recognition** 

## Writeup

[//]: # (Image References)
[image4]: ./new_images/0.png "Traffic Sign 1"
[image5]: ./new_images/2.png "Traffic Sign 2"
[image6]: ./new_images/3.png "Traffic Sign 3"
[image7]: ./new_images/4.png "Traffic Sign 4"
[image8]: ./new_images/5.png "Traffic Sign 5"

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It shows a training image and its label.


### Design and Test a Model Architecture

#### 2. Generating fake images

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

My final extended training set had 243,593 images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because my neural net was overfitting. To add more data to the the data set, I used random rotation and random translation which was done by the opencv library.


#### 3. Model architecture

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 4x4     	| 2x2 stride, validpadding, outputs 16x16x6|
| RELU					|												|
| Convolution 3x3	      	| 2x2 stride,  outputs 7x7x32 				|
| RELU	    |   									|
| Convolution 2x2		| 2x2x stride, outputs 3x3x64       									|
| flatten				|        									|dropout
|	fully connected					|	100 neurons											|
|				logits		|		43 neurons														
 


#### 4. Training

The code for training the model is located in the seventh cell of the ipython notebook. 

To train the model, I used an AdamOptimizer.
learning_rate = 0.0005
batch_size = 200
epochs = 80
beta = 0.01

#### 5. Approach

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
I began with the LeNet architecture. But due to overfitting I had to apply dropout and l2 regularization.
 

### Test a Model on New Images


#### 2. Model predictions

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| turn left ahead      		| turn left ahead   									| 
| priority road     			| priority road 										|
| road work					| road work											|
| stop	      		| Speed limit (20km/h)					 				|
| no entry			| no entry    							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 
