# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[ModelArchitecture]: ./writeup_files/ModelArchitecture.png "Model Architecture"
[LossPlot]: ./writeup_files/LossPlot.png "Validation and Training Loss"
[ImageCrop]: ./writeup_files/ImageCrop.png "Image Crop"
[Histogram_Udacity]: ./writeup_files/Histogram_Udacity.png "Histogram of Udacity Driving Data"
[Histogram_Extra]: ./writeup_files/Histogram_Extra.png "Histogram of Extra Data"
[Histogram_Recovery]: ./writeup_files/Histogram_Recovery.png "Histogram with Recovery Data"
[Histogram_Balanced]: ./writeup_files/Histogram_Balanced.png "Histogram of Balanced Data"
[Histogram_Flip]: ./writeup_files/Histogram_Flip.png "Histogram of Data after Flipping and Shifts"
[NetworkVisualisation]: ./writeup_files/FilterVisualisation.png "Visualisation of Convolutional Layers"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* finalRun_Track1.mp4 showing the models performance on track 1
* finalRun_track2.mp4 showing the models performance on track 2
* finalRun_track2_withHelp.mp4 showing the models performace on track 2 with assistance

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py MODEL_NAME.h5
```

#### 3. Submission code is usable and readable

The *model.py* file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Also *workbook.ipynb* has been included showing working throughout the project. This was werethe code development was undertaken.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based upon the NVIDIA End to End Learning model ([Link](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)). It was designed for a similar purpose it fitted well to this problem. It has 5 convolutional layers used for feature extraction and 3 fully connected layers acting as the controller.

The model uses ELU layers to introduce nonlinearity and the data is normalised using a Keras lambda layer

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers, one before the first fully connected layer and one after the first and before the second fully connected layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Initially it was tested on track 1 and once that was achieved, track 2


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Other hyperparameters such as the number of epochs were played with throughout training and selected upon the training at that stage


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of the Udacity training set, containing both positive and negative angles, 8 laps of my own collected data in both directions and recovery driving, recording of the correction of the car when getting close to the edge of the road in various situations.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to iteratively develop starting from the base NVIDIA architecture. Initially the target was to drive around track 1. This was achieved fairly early on in development when then the focus was to make create a model, trained only on data collected from track 1, drive around track 2.

My first step was to use a convolution neural network model similar to the NVIDIA End to End Model. I thought this model might be appropriate because it was developed to solve a similar, albeit real life, problem. Other models, LeNet, pre-trained VGG and self designed were considered and tried in various amounts but the initial success of the NVIDIA model meant I stuck with that. It could successfully negotiate the course only leaving the track on a couple of occassions but was also quite 'wobbly'.

The next stage was the iterative development where various ideas, such as reducing overfitting, data augmentation and preprossing, were trialed and tested out on track 1. Once the model could successfully drive the car around track 1 it was then tried on track 2. The aim was to make the model generalise well enough on  the track 1 data that it would be capable of driving track 2 without ever seeing it.

At the end of the process, the vehicle is able to drive autonomously around track 1 without leaving the road. It was also capable of driving around the majority of track 2 albeit with a little assistance on tight corners where the road was lost to the car and often on the edge of a hill. Possible improvments to the model to make it capable of completing the whole of the second track are suggested below. During development it was found there was often a trade off between the performance on track 1 and 2. If it could drive steady and in the center on track 1 it often struggled on more corners on track 2. Likewise, if it was capable of turning more corners on track 2 of would be more wobbly on track 1. In the end i opted for the latter and it provided the best performance across both tracks. The final solution still drove fairly capably on track 1 anyway.

See the included videos (LINK) for evidence of the models success on the two tracks

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

* A lambda layer performing normalisation. Input images were of a size of 66x200x3 and converted to the YUV color space as suggested by the paper. Normalisation was performed here so it could be run with GPU processing.
* 5 convolutional layers used for extracting features in the input images. The first three layers had strides of (2,2) with a 5x5 kernels. The layers consist of 24, 36 and 48 layers respectively. The final two convolutional layers consist of 64 filters each with single strides and 3x3 kernels.
* 3 fully connected layers with 100, 50 and 10 nodes. These fully connected layers are intended to act as the the controller of the system, taking the output of the convolutional filters and producing the intended steering angle
* The layer activation functions are ELU to introduce nonlinearity. This is choosen over the other activations due to its protection against a vanishing gradient and its improved speed. Theses features are found in RELU activation functions as well but ELU has been found to train faster ([Link](https://arxiv.org/abs/1511.07289))

Here is a visualization of the architecture:

![alt text][ModelArchitecture]

The following model changes and pre processing steps were done to make this possible. Data augmentations are in the follow sections.

*Dropout*

The model contains dropout layers in order to reduce overfitting. The two dropout layers are placed before the first fully connected layer and between the first and second fully connected layers. Other types of overfitting were tried such as L2 regularisation and more dropout layers. These however were found to over restrict the learning and the model was unable to learn appropriatly to make it around the whole course.

To assess the models fit the validation loss relative to the training loss is often used. However I could not get this relationship to work properly. The training loss was always higher than the validation loss during training with it often starting high and dropping through out training whilst the validation loss was lower and reduced minimally throughout training.I spend a while trying to fix this problem but was unable to.
![alt text][LossPlot]

So to assess the models fit I used its performance around the track. I found that reducing overfitting acted similarly to a filter, creating a smoother drive. Therefor if the car was found to drive 'wobbly' or in an unstable manner then methods to reduce overfitting were introduced or adjusted

Batch normalisation, something i found useful in Project 2, was tried. It is supposed to help the learning of the model and reduce the need for techniques such as dropout. However it wasn't found to be too effective in this case and so was removed.

*Number of Epochs*

After some experimentation, the number of epochs was choosen based upon the learning of the model. Utilising Keras' *ModelCheckpoint* and *EarlyStopping* callbacks only the models with the best loss were saved and the model stopped from training if the loss stopped decreasing enough. These techniques also assisted in reducing overfitting byt letting the training run for too long

By the end models fitted over about 4-6 epochs were found to perform best. It wasn't found that final validation loss was a true reflection of performance on the track however and that there was a balance between letting it train enough and getting a small loss. For example, if the model got it lowest loss on its second epoch this wouldn't be as good a model than if training was done again and the lowest loss was achieved on the fourth epoch, even if the loss is higher. This could probably be attributed to the model fitting to a local minimum.

*Image Cropping*

The image was cropped to remove the hood of the car and scenery detail above the horizon. This was done to help the model focus upon the features that I felt would be mose useful in predicting the steering angle. The amount of the crop was played with to see if any difference came about but it was found to be minimal

![alt text][ImageCrop]

*Image Input Size and Color Space*

The NVIDIA model was deisgned based upon images sized at 66x220 and in the YUV plane. The image size was adjusted to make the model train faster and wasn't seen to have any detrimental effects on the models driving ability. However the final convolutional layer was only a 1x1 size and i felt that this could be potentially limiting going further in development so was changed back.

The color space wasn't played with but one option would have been to have an initial 3x3 fully connected layer after the normalisation layer to let the model pick the colors from the images it found most relevant. This could be a future work.

#### 3. Creation of the Training Set & Training Process

*Initial Dataset*

All initial model development was completed using the provided Udacity dataset. This was a really good dataset to use initially due it being well balanced with a good range of both positive and negative driving angles. This helped get confirm the model and gain various understanding about the project. One of the main bits of knowledge gained was that it was better to have less well balanced data than more unbalanced data. This was becasue more unbalanced data, such as a disproportionate number of zero angle samples, could 'distract' the model and cause it to overfit these smaller angles.

![alt text][Histogram_Udacity]

*Extra Data Recording*

The NVIDIA dataset was suitable enough to get the model to complete track one. However the model could not complete track 2 no matter what changes I made to the model architecture. To get more data to for the model to train on I recorded 4/5 laps in each direction around track 1. This had the effect of making the driving on track 1 more stable and improved the models ability on track 2. Making it up the first hill and around the corner it often drove striaght on previously. Once i hit the shadow of the hill it was unable to find features due to not having seen dark patches previously. For now the simulator was run at a lower graphic quality to remove the shadows so to better assess the models ability to handle the corners.

![alt text][Histogram_Extra]

*Recovery Driving*

The dataset was lacking in larger steering angles and this reflected in the model which was incapable of turning sharper corners. To was remeded by recording 'recovery' and generally erratic driving to produce some larger angles for the dataset. Furthermore the left and right images for each sample were loaded and their driving angles derived by shifting the angle from the center camera. This all produced the following distribution.

![alt text][Histogram_Recovery]

*Balancing Dataset*

However, despite the extra data it was not capable of getting further on track 2. This is where the knowledge gained earlier came in. The model contained too many small angles meaning the model would generally favour them rather than the larger angles needed to turn the sharper corners on track 2. This was solved by randomly removing 98% of values less than 0.375º and 60% of values less than 2.5º each epoch. This produced the following distribution.

![alt text][Histogram_Balanced]

The model was then capable of getting up the first hill and then got stuck on the first sharp corner, which was awesome! This did however have a detrimental effect on track 1. With the model more pre disposed to large steering angles the car drive more erratic and crossed the edges of the road a couple of times. Smoothing of the driving was done by  trying various methods to prevent overfitting

The dataset was still fairly unbalanced with few driving angles in between ~0.25 and 1.0. To produce more of these angles the follow data augmentations were done to images with angles between 0.3 and 0.9:
* **Image Flip** - Images and their driving angles were flipped
* **Image Shift** - Images were randomly shifted by up to +/-10 pixels horizontally and vertically.  When shifted horizontally the driving angle was also adjusted to simulate a new angle. This was done twice for each image

These changed produced the following distribution:

![alt text][Histogram_Flip]

*Brightness and Shadows*

As track one contained minimal shadows and was of a continous brightness balanced this meant the car was unable to drive in shadowed and dark areas on track 2. To try to mimic these conditions with track 1 data, random shadow (using code from [Vivek Yadev](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)) and random adjustments to the image brightness were made to the exsisting data set. This lead to the car being really stable on track one but performance of track 2, both with and wihtout shadows, was worsened. These augmentations were removed in favour for the better performance of the previous model on track 2. However future work, possibly reducing the amount of brightness augmentation, should include implementation of these augmentations as, if done properly, should be beneficial to the models performance.

*Final Performance*

After all of the augmentations and balancing of the data set the model was suitable of driving both track 1 and track 2 (with a little help). After this various amounts of Dropout was tried to help reduce overfitting and make the driving less erratic on track 1 whilst not effecting the performance on track 2. This produced the final model that can be seen in VIDEO (LINK!!!)

VIDEO OF MODEL??

*Visualisation of the Filters*

To get an insight the the 'black box' of the network I visualised the output of the convolutional layers of the model. These can be seen below. It can be seen that the layers pull the edges and lines from the images in the first two layers. The following layers become less coherent but it seems that they are breaking the images down to what it sees as the most important features resulting in a 8 bit layer for the fully connected layers to process upon

![alt text][NetworkVisualisation]

*Problems with the model*

The model is unable to handle the following areas on track 2:
* Sharp U-turns on hill edges
* Places where the road is not visible due to it turning down hill

Both of these problems are thought to be due to a mix of the car losing sight of the road and therefore not knowing which way to go and the road leaving to the side of the image, something seen minimally from track 1. The fix for this would be to try and replcate these scenarios by further manipulating the data from track 1. Some more augmentation ideas could be warping the images to mimic a sharp turn to the side of the image or the road being lost beneath the brow of a hill. This would then give the model a better ability to predict the steering angle as it loses sight and then regains sight of the road.


After the collection process and data augmentation I had _________ number of data points to train on. The data was shuffled each epoch to limit any accidental pattern from the order of the images. Training was done uing an adam optimizer so that manually training the learning rate wasn't necessary and the rate of learning was improved.
