## Project Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier.
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/predictions_1.png
[image4]: ./output_images/predictions_2.png
[image5]: ./output_images/predictions_3.png
[image6]: ./output_images/predictions_4.png
[image7]: ./output_images/predictions_5.png
[image8]: ./output_images/predictions_6.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

This step is contained in VehicleDetection.VehicleClassifier, in the method train(). First I specify a list of directories to read training images from. This method then passes on the list of directories to the train() method of the VehicleClassifier class. The first two steps in this method getRawData() and extractFeatures() load the training images, which consist of `vehicle` and `non-vehicle` images. Here is an example of each one of these:

![alt text][image1]

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).   Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of HOG parameters to train my classifier, and settled on the one that gave better score on the test set. Besides the test set from train_test_split, I also tested the classifier on single frames extracted from the test video. For orientations, higher than 9 was giving worse results so I settled at 9. That number was also suggested as a inflection point in the HOG paper.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The step to train a classifier is contained in package VehicleDetection, class VehicleClassifier, method train(). For the features, I used HOG, color histogram and spatial features. The code for these was taken from the Udacity quiz file lesson_functions.py (also contained in package VehicleDetection). In the train() method, after loading the car and non-car images, I split the data into train/test sets using train_test_split() method of sklearn.model_selection. I then fitted a Standard_Scaler on the training data and kept it around for testing and future processing. The scaled training data was then fed to AdaBoost with Linear SVM as the base classifiers.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in package `VehicleDetection`, class `VehicleClassifier`, method `find_cars()`. This method is borrowed from the Udacity quiz, however there are some changes to it. The method takes in the window scale to search (the base window is 64x64 pixels), and also a range of y-axis to search between, and it returns a list of window coordinates that are predicted to be cars. This method first extracts the HOG features from the entire frame, then selects a window based on scale and other parameters. The method is called from `VehicleClassifier.identifyVehicles`, which passes various search scales to it. The search scales were selected based on performance on snapshots taken from the test video, as well as clips taken from the project video. Initially I used 2 scales of 1.5 and 2.0. This was doing well for the cars that are near-by, but I wanted to cover a little more distance. So I ended up using 3 scales, `[1.0, 1.5, 2.5]`, with the following ranges of y-axis respectively: `[350,500], [350,512], [400,680]`

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some example images:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

I first used a plain Linear SVM classifier, but it was producing a lot of false positives. I then did negative sample mining to extract false positives from a few frames of the test video and fed them as non-car training examples. I had to repeat this train/test cycle a number of times and feed false positives from one cycle to the next. This led me to consider AdaBoost. It uses an ensemble of weak classifiers, where the misclassified samples from one classifier are assigned a higher weight to train the next classifier. I used Linear SVM as the base classifier for AdaBoost, and with good results on the test video, I settled on this choice. 

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
