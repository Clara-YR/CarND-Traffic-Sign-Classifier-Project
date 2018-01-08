# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/labels_distribution.png "Visualization"
[image2]: ./examples/rgb_gray.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/new_images.jpg "Traffic Signs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Clara-YR/CarND-Traffic-Sign-Classifier-Project/blob/origin/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is __34799__.
* The size of the validation set is __4410__.
* The size of test set is __12630__.
* The shape of a traffic sign image is __(32, 32, 3)__.
* The number of unique classes/labels in the data set is __43__.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale in order to descriminate the image shape __from (32, 32, 3) to (32, 32, 1)__.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data to convert the pixels range __from [0, 255] to [-1, 1]__.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution_1 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU_1					|												|
| Max Pooling	_1      	| 2x2 stride,  outputs 14x14x6				|
| Convolution_2 5x5    | 1x1 stride, valid padding, outputs 10x10x16    									|
| RELU_2		|        									|
| Max Pooling_2			| 2x2 stride, outputs 5x5x16       									|
| Flatten					|	outputs 400											|
|Fully Connected_1					|outputs 120						|
|RELU_3					|    |
|Fully Connected_2|  outputs 84|
|RELU						|   |
|Fully Connected_3|  outputs 43|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:

- EPOCHS = 20
- BATCH_SIZE = 40
- learning_rate = 0.002
- optimizer - `tf.train.AdamOptimizer()`

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of __0.934__.
* validation set accuracy of __0.917__.
* test set accuracy of __0.833__.

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen? 
  __The first archtecture was tried to pick out simple shapes and patterns sucn as diagonal lines and color blobs.__
* What were some problems with the initial architecture?
__The output deepth maybe not enough for complicated images.__
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting. __At first I tried to change the variance of the output depth from 1 - 6 - 16 - 400 - 120 - 84 - 43 to 1 - 10 - 20 - 500 - 240 - 168 - 43, but it makes no big contribution to the accuracy output, so I finally keep the network the same structure  as LeNet 5.__
* Which parameters were tuned? How were they adjusted and why?
__I tuned EPOCHS from 2 to 20 to ensure enough times for the network to learn and modify its weights in order to get a accuracy bigger the 0.930, at the same time I tune the BATCH_SIZE from 128 to 64 and the learning rate from 0.001 to 0.002 to make the neural network to learn faster.__
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
__Convolution layer is goot at pick out image features. Dropout layer can reduce data loss comparing with max pooling layer.__

If a well known architecture was chosen:

* What architecture was chosen?
__I choose the LeNet 5 architecture.__
* Why did you believe it would be relevant to the traffic sign application?
__Because LeNet 5 works well on images classification.__
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
__The accuracy on the training, validation and test set are all above 0.90__
 

### Test a Model on New Images

#### 1. Choose six German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4]

The second and fifth images might be difficult to classify because they are too dark.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 17 - No entry|17 - No entry| 
| 8 - Speed limit(120km/h)| 8 - Speed limit(120km/h)|
| 34 - Turn left ahead|34 - Turn left ahead|
| 38 - Keep right| 12 - Priority road|
| 0 - Speed limit(20km/h) | 0 - Speed limit(20km/h) |
|28 - Children crossing|28 - Children crossing|

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of __83.3%__. This compares favorably to the accuracy on the test set of __91.5%__.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th and 20th cells of the Ipython notebook.

---
For the first image, the model is relatively sure that this is a __no entry__ sign (probability of 1.00), and the image does contain a __no entry__ sign. The top five soft max probabilities were

| Probability  1       	|     Prediction	1        					| 
|:---------------------:|:---------------------------------------------:| 
|1.0000000000|17|
|0.0000000000|38|
|0.0000000000|14|
|0.0000000000|37|
|0.0000000000|34|
---
For the second image, the model is relatively sure that this is a __speed limit (120km/h)__ (probability of 1.00), and the image does contain a __speed limit (120km/h)__ sign. The top five soft max probabilities were

| Probability  2       	|     Prediction  2	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.0000000000|8|
|0.0000000030|7|
|0.0000000000|4|
|0.0000000000|1|
|0.0000000000|0|
---
For the third image, the model is relatively sure that this is a __turn left ahead__ sign (probability of 1.00), and the image does contain a __turn left ahead__ sign. The top five soft max probabilities were

| Probability  3       	|     Prediction  3        					| 
|:---------------------:|:---------------------------------------------:| 
|1.0000000000|34|
|0.0000000000|38|
|0.0000000000|32|
|0.0000000000|3|
|0.0000000000|12|
---
For the fourth image, the model tend to judge this is a __no entry__ sign (probability of 0.71), however the image actually contains a __keep right__ sign. The top five soft max probabilities were

| Probability  4       	|     Prediction  4        					| 
|:---------------------:|:---------------------------------------------:| 
|0.7078289390|12|
|0.2921677530|1|
|0.0000026338|11|
|0.0000006988|25|
|0.0000000067|38|
---
For the fifth image, the model is relatively sure that this is a __speed limit (20km/h)__ sign (probability of 0.99), and the image does contain a __speed limit (20km/h)__ sign. The top five soft max probabilities were

| Probability  5       	|     Prediction  5        					| 
|:---------------------:|:---------------------------------------------:| 
|0.9879285690|0|
|0.0120713720|1|
|0.0000000733|8|
|0.0000000005|7|
|0.0000000002|4|
---
For the sixth image, the model is relatively sure that this is a __children crossing__ sign (probability of 0.99), and the image does contain a __children crossing__ sign. The top five soft max probabilities were

| Probability  6       	|     Prediction  6        					| 
|:---------------------:|:---------------------------------------------:| 
|0.9999998810|28|
|0.0000000626|29|
|0.0000000003|5|
|0.0000000000|3|
|0.0000000000|19|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


