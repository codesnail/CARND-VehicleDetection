## Vehicle Detection

The goal of this project is to identify vehicles on the road in a video feed. This is done by training a classifier to identify cars, then implementing a pipeline to feed video stream of a road to the classifier and using it to detect cars.

The project implements the following pipeline:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a AdaBoost classifier.
* Use color and spatial features. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image1.1]: ./output_images/cars.png
[image1.2]: ./output_images/not_cars.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/predictions_1.png
[image4]: ./output_images/predictions_2.png
[image5]: ./output_images/predictions_3.png
[image6]: ./output_images/predictions_4.png
[image7]: ./output_images/predictions_5.png
[image8]: ./output_images/predictions_6.png
[image9]: ./output_images/heat_map1.png
[image10]: ./output_images/int_heat_map1.png
[image11]: ./output_images/heat_map2.png
[image12]: ./output_images/int_heat_map2.png
[image13]: ./output_images/heat_map3.png
[image14]: ./output_images/int_heat_map3.png
[image15]: ./output_images/heat_map4.png
[image16]: ./output_images/int_heat_map4.png
[image17]: ./output_images/heat_map5.png
[image18]: ./output_images/int_heat_map5.png
[image19]: ./VehicleDetection_Classes.jpg
[image20]: ./VehicleDetection_Sequence.jpg
[video1]: ./project_video_out.mp4

### Project Design

#### Class Diagram
The following class diagram shows a structural view of the classes making up the pipeline:

![alt text][image19]

#### Sequence Diagram
The following diagram shows the sequence of how the pipeline is executed:

![alt text][image20]

1. `VehicleDetection.VehicleClassifier.py`: Contains the class VehicleClassifier with methods train(), extractFeatures(), find_cars(), identifyVehicles() etc. This python file also contains some methods defined outside of the class, such as `getSavedVehicleClassifier, train(), testScanImage(), testIndividual()` for getting a saved classifier, initiating training and saving a classifier to disk, and testing and troubleshooting individual frames and images.
1. `run_vehicle_detection2.py`: Main program that initiates and runs the pipeline. This program uses a saved classifier that is trained by the first program above, by calling the `getSavedVehicleClassifier()` method defined in `VehicleDetection.VehicleClassifier.py`. It reads through the video frame by frame, and calls `VehicleClassifier.identifyVehicles()` on each. It also aggregates the heatmap and draws bounding boxes around identified vehicles.
1. `VehicleDetection.lesson_functions.py`: Utility module that contains methods to extract hog and other features.
1. `VehicleDetection.heat_map.py`: Utility mdodule that contains methods to create heatmap of detected vehicles.
1. `classifier4.pkl`: Pickle file containing the trained classifier and scaler used for this submission.


### Data Exploration

The training data for this project consists of cars and non-car images seen on roads. Here is an example of `vehicle` and `non-vehicle` images:

![alt text][image1]

The image dimensions are 32x32x3. All images are taken from behind the vehicles, some directly behind and some from an angle. All `vehicles` are cars of various models and colors. No trucks or buses are included in this project.

### Feature Extraction

For the features, I used HOG, color histogram and spatial features. The method extractFeatures() in VehicleClassifier generates these features from images. I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

I settled on the parameters that gave better score on the test set. Besides the test set from train_test_split, I also tested the classifier on single frames extracted from the test video. For orientations, higher than 9 was giving worse results so I settled at 9. That number was also suggested as a inflection point in the original HOG paper.

### Training and Classifier Choice

The step to train a classifier is contained in package VehicleDetection, class VehicleClassifier, method train(). After loading the car and non-car images, I split the data into train/test sets using train_test_split() method of sklearn.model_selection. I then fitted a Standard_Scaler on the training data and kept it around for testing and future processing. The scaled training data was then fed to AdaBoost with Linear SVM as the base classifiers.

### Sliding Window Search

The sliding window search is implemented in package `VehicleDetection`, class `VehicleClassifier`, method `find_cars()`. The method takes in the window scale to search (the base window is 64x64 pixels), and also a range of y-axis to search between, and it returns a list of window coordinates that are predicted to be cars. This method first extracts the HOG features from the entire frame, then selects a window based on scale and other parameters. The method is called from `VehicleClassifier.identifyVehicles`, which passes various search scales to it. The search scales were selected based on performance on snapshots taken from the test video, as well as clips taken from the project video. Initially I used 2 scales of 1.5 and 2.0. This was doing well for the cars that are near-by, but I wanted to cover a little more distance. So I ended up using 3 scales, `[1.0, 1.5, 2.5]`, with the following ranges of y-axis respectively: `[350,500], [350,512], [400,680]`

### Example Results and Optimization

Here are some video frames with example results:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

I first used a plain Linear SVM classifier, but it was producing a lot of false positives. I had to do significant negative sample mining to extract false positives from a few frames of the test video and fed them as non-car training examples. I had to repeat this train/test cycle a number of times and feed false positives from one cycle to the next. This led me to consider AdaBoost, since it uses an ensemble of weak classifiers, where the misclassified samples from one classifier are assigned a higher weight to train the next classifier. I used Linear SVM as the base classifier for AdaBoost, and with good results on the test video. 

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The filteration code is implemented in the main program that runs the pipeline: `run_vehicle_detection2.py`. This program loops through the video frames and calls `VehicleClassifier.identifyVehicles()` on each frame. It returns a heatpmap of detected pixels for each frame. These heatmaps are stored in a list of 5 consecutive frames. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the aggregated heatmaps. For aggregation, I first tried taking the mean of the heat maps and thresholding at 3 detections per frame. This worked pretty well to filter out the false positives, but it doesn't create very good bounding boxes and is late in identifying vehicles that are just appearing in the frame. So I used the sum of 5 subsequent heatmaps and thresholding them at 12 total detections (factoring in that I'm searching on 3 scales). I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video. In my implementation, I integrate and threshold the heatmap after each frame. The advantage is that the vehicle identification starts early. In the images below, each individual frame's predictions and heatmap is shown, followed by the integrated heatmap of the last 5 frames upto that point after thresholding:

#### Frame1:

![alt text][image9]
![alt text][image10]

The first integrated heatmap is empty as the required threshold is not yet met.

#### Frame2:
![alt text][image11]
![alt text][image12]

The second integrated heatmap consisting of the first 2 frames meets the threshold, and the car positions are displayed.

#### Frame3:
![alt text][image13]
![alt text][image14]

#### Frame4:
![alt text][image15]
![alt text][image16]

Here, we can see the false positive prediction is removed in the integrated heatmap.

#### Frame 5:
![alt text][image17]
![alt text][image18]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first problem I faced was the detection of lots of false positives. I already discussed the approach I took above where I discussed how I optimized the classifier using negative mining and using AdaBoost. To me the major challenge of this approach is to identify non-cars. The current pipeline still incorrectly classifies some sign-boards and probably will fail on other objects not explicitly trained on that look similar to a car. I can remove these by targeted negative mining and training on them. For me this is the major pitfall of this approach, i.e., it relies on negative mining, and so makes it a bit unreliable. Ideally I would like to explore more features to identify the positive examples more correctly. I could use template matching, but the template has to be generic enough or I will have to use multiple templates. 

Another approach would be to implement a bayesian predict/sense cycle, with a Kalman or particle filter. But this would work only after a vehicle is detected initially, then we can follow it more accurately by updating its sensed position with a motion prediction model.

I would like to try deep learning to see if that works better, however to me it's lack of an explanable model would be a key concern from safety perspective (but similar concern is present in the current approach).

Another thing I want to try is combining edge detection in the pipeline. Perhaps identifying object boundaries and then feeding the objects into the classifier rather than a blindly sweeping search window.
