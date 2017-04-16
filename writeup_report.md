**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## Rubric Points 
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation. 
---

###Histogram of Oriented Gradients (HOG), Histogram of Colors and Spatial Binned

The code for this step is contained in the 1st, 2nd, 3rd and 4th code cells of the iPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then created functions to extract the spatial binned features, the histogram of colors and the HOG features - 7th cell of iPython notebook. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed a sample images from each of the two classes and displayed them to get a feel for what each feature vector looks like.

Here is an example using the histogram of colors, spatial binned and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

After I tried various combinations of parameters, I settled with 12 orientations, 8x8 pixels per cell and 2x2 cells per block for the HOG features, 32x32 image size for the binned color features and 32 bins for the histograms, which provided the best results. Those parameters are set on function called `tunning_parameters()` located at the 8th cell of the notebook.

I trained a linear SVM using the LinearSCV() function to classifiy images as vehicles. The scv is fit with a training set that is obtained from a combination of the normalized vehicles and non vehicles features vectors. My classifier gets an accuracy of 98.%. The classifier is located at the 9th cell of the notebook.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I impletemented 2 different functions to handle the sliding window search. A function called `slide_window()`, located at the 6th cell of the notebook, is used to simply show the windows that will be searched throught the frames. A second function, which is part of the final pipeline, called `find_cars()`, located at the 13th cell of the notebook, is a more efficient method for doing the sliding window approach. This function only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. I used a scaling factor of 1.5, with cells_per_step = 2, which resulted in a serch window overlap of 50%. 

![alt text][image3]

Ultimately I searched on one scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is an example image:

![alt text][image4]
---

### Video Implementation
Here's a [link to my video result](./project_video.mp4)

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

