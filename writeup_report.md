**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/example_car_noncar.png
[image2]: ./output_images/car_binned_histogram.png
[image3]: ./output_images/car_hog.png
[image4]: ./output_images/noncar_binned_histogram.png
[image5]: ./output_images/noncar_hog.png
[image6]: ./output_images/slide_window_region.png
[image7]: ./output_images/find_cars.png
[image8]: ./output_images/full_pipeline.png
[image9]: ./output_images/final_result.png

[video1]: ./project_video.mp4

## Rubric Points 
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation. 
---

###Histogram of Oriented Gradients (HOG), Histogram of Colors and Spatial Binned Colors

The code for this step is contained in the 1st, 2nd, 3rd and 4th code cells of the iPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then created a function called `extract_features()` to extract and combine the spatial binned features, the histogram of colors and the HOG features - 6th cell of iPython notebook. I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) as well as different image sizes for binned colors and diferent number of bins for the histrogram.  I grabbed a sample image from each of the two classes and displayed them to get a feel for what each feature vector looks like.

Here is an example using the histogram of colors, spatial binned and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, `spatial_size=(32,32)` and `history bins=32` individually:

![alt text][image2]
![alt text][image3]

![alt text][image4]
![alt text][image5]

After I tried various combinations of parameters, I settled with 9 orientations, 8x8 pixels per cell and 2x2 cells per block for the HOG features, 32x32 image size for the binned color features and 32 bins for the histograms, which provided the best results. Those parameters are set on function called `tunning_parameters()` located at the 11th cell of the notebook.

I trained a linear SVM using the `LinearSCV()` function to classifiy images as vehicles. The scv is fit with a training set that is obtained from a combination of the normalized vehicles and non vehicles features vectors. My classifier gets an accuracy of 98.73%. The classifier is located at the 12th cell of the notebook.

###Sliding Window Search

I impletemented 2 different functions to handle the sliding window search. A function called `slide_window()`, located at the 9th cell of the notebook, is used to simply show the windows that will be searched throught the frames. A second function, which is part of the final pipeline, called `find_cars()`, located at the 14th cell of the notebook, is a more efficient method for doing the sliding window approach. This function only has to extract hog, histogram and binned color features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. I used a scaling factor of 1.4, with cells_per_step = 2, which resulted in a serch window overlap of ~55%. The `find_cars()` function uses the linearSVC classifier to make the predictions.

![alt text][image6]

Ultimately I searched on one scale (1.4) using YCrCb 3-channel HOG features combined with binned color and histograms of colors feature vectors, which provided a nice result.  Here is an example image:

![alt text][image7]
---

### Video Implementation
Here's a [link to my video result](./project_video.mp4)

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The pipeline for the video implementation is located at the 18th cell of the notebook.  


### Here is an example of the results of my pipeline in a sample image:

![alt text][image8]


---

###Discussion

The pipeline has a a bit of a hard time identifying the white car across all the frames. It also finds a few too many false positives, which makes me believe that light intensity and shadows are interfering with the results. The results could be potentially improved by exploring different color spaces, updating the number of previous frames, tunning the threshold and the vector features parameters. I also strongly believe that a neural network would have a far better result than a LinearSVC. 

