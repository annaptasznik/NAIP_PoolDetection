# NAIP Pool Detection

## Introduction

This work demonstrates a method for identifying features in NAIP aerial imagery. Modelled as an object recognition problem, a CNN is used to identify images as being swimming pools or something else--specifically a street, rooftop, or lawn.

This work was completed for my graduate computer vision course in Autumn 2019. Though it was designed to identify swimming pools, it can be extended to identify any other desired feature. Shapefiles representing feature locations are accepted to generate training samples and train a CNN, which in turn identify similar features in new regions. A python package for this is coming in Winter 2020.


## Related Work

Convolutional neural networks (CNNs) are used in a wide variety of studies to identify features in satellite data. Some approaches are quite advanced, such as using the SIMRWDWN framework and a 22 layer deep network [8], Random ZFNet_Forest methods combined with CNNs [2], or CNNs across the temporal dimension (TempCNNs) [6]. Researchers have used segmentation techniques, object detection techniques, or combinations of both [4].

I opted to design a simple CNN like those outlined by Zhong et al. [9]. Unlike Zhong et al. and other studies, I will also be preparing my own image samples for testing. The reasoning for this is twofold: because 1) these tools can be easily extended to other features, and 2) common image libraries like SAT-4, SAT-6, SpaceNet, DOTA, or COWC do not have categories for swimming pools. For this part of my model, my approach is like Albert et al. [1], who used the Google Maps Static API to obtain images. The NAIP imagery product is similar in that it is free and publicly available but differs because there are no daily limits to the number of pictures that can be obtained. NAIP imagery is a common imagery product used in remote sensing studies.
Since the best CNN results often require fine tuning the model [3], I also introduce a basic way to choose parameters based on accuracy success. Similar methods have been employed for finding image transformation parameters [7].

## Method
### Overview
At a high level, this problem was addressed in five steps. First, I transformed geographic information into a collection of image classes. Then, I cleaned the image classes to be of the same size and resolution as one another. Once image classes were prepared, I wrote a simple CNN to learn from training images and create a model for predicting the classification of new images. Then, I used an automated process of trial-and-error to determine optimal CNN parameters and used them to generate a model for evaluating testing data. Finally, the model was used to identify features in other sets of imagery.

<p align="center">
  <img src="https://github.com/annaptasznik/NAIP_PoolDetection/blob/master/bin/images/project_images/fig1.PNG">
</p>


### Machine Specification
This analysis was performed on a Microsoft Surface Book running Windows 10. It has 8GB of installed RAM and an Intel® Core™ i5-6300U CPU @ 2.40 GHz 2.50 GHz processor. 

### Software

All steps of the process were written using Python. Python was chosen because of its ease of use, abundance of reliable modules, and compatibility with common geographic information systems. Specifically, I used the following libraries:


| Python Library | Primary Use |
| :------------- | :----------: | 
|  arcpy | Handling spatial information  | 
| PIL | Basic image manipulation | 
| pytorch and torchvision | Creating CNN | 
| numpy | Handling image arrays | 
| datetime | Measuring time elapsed in functions | 
| tqdm | Visualizing epoch training progress | 
| matplotlib | Creating plots and visualizing images | 
| os | Moving, deleting, and creating files | 
| csv | VWriting results to a table | 	
	
Additionally, I used ESRI’s ArcGIS suite, a widely-used geographic information system, to handle spatial data. This was critical for transforming geospatial information (points on a map) to images that could used in a classic computer vision problem. More details are described on this below.

### NAIP Image Data Source

The National Agricultural Imagery Program (NAIP) is an aerial survey conducted every three years over the United States. Image resolution is 1m and captured in RGB image bands, though some recent surveys include the near infrared band as well. Since an average swimming pool is several square meters in area, NAIP resolution was enough to adequately capture the features. Additional benefits to NAIP imagery are that all images are cloud-free (a common issue while working with satellite images) as well as free and publicly available.

<p align="center">
  <img src="https://github.com/annaptasznik/NAIP_PoolDetection/blob/master/bin/images/project_images/fig2.PNG">
</p>


### Finding Class Points Using Geographic Information Systems 

Leveraging the abundance of publicly available geospatial data made it possible to automate much of the training and test image creation. 

### GIS Software and Data Sources

Geographic information systems (GIS) are a subset of information systems specifically designed for spatial data. I used GIS to prepare training and testing images.
Though many software packages exist to work with geospatial data, I used a common and industry-standard GIS software called ESRI ArcGIS. In addition to having the ability to view and build maps, the ArcGIS suite makes spatial functions and data types available through a python package called arcpy. Since there is a user license associated with arcpy (it is not available through pip or other package managers), all functions using arcpy are contained within the file titled geo_utils.py for ease of use to those who do not have a software license.
GIS data is often available online from municipalities. I used some such data from the City of Phoenix to obtain the city boundary. Since the information in these data sources is limited and varies broadly across different municipalities, I also used open source spatial data from Open Street Map (OSM), an open source and community-sourced platform for spatial data. In OSM, there is data for streets, public places, electrical infrastructure, schools, libraries, parks, and a myriad of other places. 

### Spatial Data for Building Image Classes

I selected four image classes—lawn, street, roof, and pool—and collected samples of each class from metropolitan areas in Arizona and Southern California (Mesa, Tucson, Chandler, Tempe, San Diego, Orange County, etc.). These areas were selected for training images because they contain many residential swimming pools and are closest to the testing area of Phoenix, Arizona.

### Pool Class

Swimming pools in OSM are public pools and are much larger than residential pools, so I did not believe they alone would constitute a full training set. Thus, I made my own pool data by manually identifying residential pools and creating a point feature class for each one in ArcGIS. 

<p align="center">
  <img src="https://github.com/annaptasznik/NAIP_PoolDetection/blob/master/bin/images/project_images/fig3.PNG">
</p>


### Lawn Class

Lawn points were identified by randomly selecting points within park and sports field features. However, to capture residential lawns, I also manually identified and created a point feature class for each one in ArcGIS. 

### Roof Class

Roof points were identified using US Building Footprints generated by Bing [7] based on a method developed by RefineNet (a method for connecting segmented images) [4]. I randomly selected the centroids of building footprints under the size 2,000 m^2. This selection process was confined to specific areas and controlled for quality manually. 

### Street Class

Street points were identified using OSM’s streets feature. I randomly selected points within a 10m buffer around street center lines.

<p align="center">
  <img src="https://github.com/annaptasznik/NAIP_PoolDetection/blob/master/bin/images/project_images/fig4.PNG">
</p>

### Image Preparation

Much preprocessing was necessary to extract CNN-ready images from raw NAIP imagery. Once points of interest were created in a file geodatabase feature class, a series of scripts were written to export a bounding box of NAIP imagery snippets. Once all images were exported, they were formatted into sizes of 30x30 pixels. All the image preparation tools can be found in utils.py.
 
<p align="center">
  <img src="https://github.com/annaptasznik/NAIP_PoolDetection/blob/master/bin/images/project_images/fig5.PNG">
</p>


### Convolutional Neural Network (CNN)
I used a CNN built in PyTorch. My model was largely built from Beibin Li’s example from class. The CNN is a process of two 2D convolution layers, 1 max pooling layer, and 1 fully connected layer.

 
<p align="center">
  <img src="https://github.com/annaptasznik/NAIP_PoolDetection/blob/master/bin/images/project_images/fig6.PNG">
</p>

### Automatic CNN Parameters

Choosing optimal CNN parameters relies on a process of trial and error. I automated the choice of specific CNN parameters, epoch and batch size, according to which parameters yielded in the best training accuracy. 

The script get_best_params.py runs CNN training multiple times with varying epoch and batch sizes, saving the most accurate training model in the scripts results directory. 


## Experiments and results

### Evaluation Criteria
My primary focus is to achieve a testing accuracy above 90%. Thus, testing accuracy is my criteria for evaluating the success of my model.

### Performance
Upon finding the optimal parameters for my CNN (epoch and batch size), I found CNN model training to consistently take less than 3 minutes. When optimizing my parameters through my script detailed below, epoch_size and batch_size were both very low, thus contributing to the low training times.

### Results

#### Training and Testing Images
The aforementioned NAIP extraction process yielded samples for each image class—street (n_train = 434, n_test = 227), lawn (n_train = 410, n_test =139), pool (n_train = 703, n_test = 594), and roof (n_train = 670, n_test = 222). A sample of each class is shown below.

 
<p align="center">
  <img src="https://github.com/annaptasznik/NAIP_PoolDetection/blob/master/bin/images/project_images/fig7.PNG">
</p>

I found the average histogram of RGB values for each image class using the custom get_RGB_freq() function in utils.py. As expected, we see that each image class has a distinct color distribution. Lawns, pools, and streets have more pixels with high green and blue values, while roofs have more reds.
 
<p align="center">
  <img src="https://github.com/annaptasznik/NAIP_PoolDetection/blob/master/bin/images/project_images/fig8.PNG">
</p>

To confirm the degree to which training sample size affects CNN accuracy, I did a quick test to determine accuracies with different sample sizes.

 
<p align="center">
  <img src="https://github.com/annaptasznik/NAIP_PoolDetection/blob/master/bin/images/project_images/fig9.PNG">
</p>

As expected, accuracy generally gets better with larger sample sizes. 

#### Testing Individual CNN Parameters

Before automating CNN parameter selection, I ran a few tests to understand the effects of epoch, batch size, and shuffling on training and test accuracy. 
Training and testing accuracy improved with more epochs, with the largest accuracy gains being made in 0 – 50 epochs. Thereafter, accuracy gains were marginal. This result was expected because greater epochs mean greater iterations of passing the dataset forward and backward through the neural network.

 
<p align="center">
  <img src="https://github.com/annaptasznik/NAIP_PoolDetection/blob/master/bin/images/project_images/fig10.PNG">
</p>

What was found is that shuffling samples with every epoch (shuffle = True) makes a hug difference in training and testing accuracy—at least 3x more accurate. This finding makes sense because shuffling samples reduces the chances of the CNN fitting towards noise in the image classes. Due to this finding, I chose shuffling to be a requirement of my CNN.
The effect of batch size on testing accuracy was varied with no distinct trend. This might have been different had I tried batch sizes that were the size of my entire sample, however, due to CPU limitations I kept my experiment to small batch sizes. Since the result is unpredictable, it became clearer that an automated approach to determining batch size would be ideal, as the “sweet spot” could be found. 


<p align="center">
  <img src="https://github.com/annaptasznik/NAIP_PoolDetection/blob/master/bin/images/project_images/fig11.PNG">
</p>

	
#### Optimizing CNN Parameters
Testing the accuracy of different epoch and batch size permutations determined an optimal parameter of epoch_size = 10 and batch_size = 3. Note, the permutations were limited such that computation time and CPU utilization was not unreasonably high. These results are only optimized for my specific computational and time context.

#### Success in Pool Detection
	
Using the parameters deemed optimal by the get_best_params.py script, I found an average testing accuracy of 83%. The highest observed testing accuracy was 88%.

| Trial	| TestAcc |	TrainAcc|
|1	|0.883928571	|0.918010753|
|2	|0.810876623	||0.959229391|
|3	|0.756493506	|0.943100358|
|4	|0.836850649	|0.9359319|
|5	|0.850649351	|0.961021505|
|6	|0.841720779	|0.966845878|
|7	|0.861201299	|0.959677419|
|8	|0.815746753	|0.943996416|
|9	|0.866071429	|0.958333333|
|10	|0.818993506	|0.954749104|
|Average |	0.834253247	| 0.950089606|

Digging deeper into the misidentified images sheds some insight on what could be done in future iterations of this work.

<p align="center">
  <img src="https://github.com/annaptasznik/NAIP_PoolDetection/blob/master/bin/images/project_images/fig13.PNG">
</p>

We can see from the confusion matrix that most misidentifications are mistaken for roofs. This is likely due to the variation in roof samples, particularly in color. Perhaps I would have found more success had different colored roofs been separated into different classes. 
 
## Future work

There are several ways to both improve upon this project as well as to extend its value.

The existing project could be improved upon using several more advanced CNNs such as r-CNN or a combination of segmentation and object detection. Image classes and number of samples could be increased, which would increase accuracy and applicability across different and new geographies. Sample size and image classes could be increased by leveraging existing satellite imagery identification projects such as SpaceNet, as well as expanding upon the process for creating samples from NAIP imagery. In the latter, the extent of data collection could be expanded to include samples across new locales. Additionally, the use GPUs for training would allow the processing of larger datasets and more complicated models.

Additionally, a large value add would be in creating an API which would allow a user to input latitude and longitude points and identify the object at that site on the fly. Such functionality would increase the ease of use to those who want to understand their own regions of interest.


## Summary and Conclusion

In this project, I have demonstrated a workflow for creating enough training data from satellite imagery and creating a model which predicts the classes of test data at an accuracy of over 88%. Though there are many avenues by which the accuracy could be improved, my process has leveraged free satellite imagery and a basic commercial laptop to perform a full object detection workflow, demonstrating the potential to get reasonable accuracy with few special resources. 


## References

[1] Albert, A., Kaur, J., and Gonzalez, M. (2017). Using Convolutional Networks and Satellite Imagery to Identify Patterns in Urban Environments at a Large Scale, Cornell University, https://arxiv.org/pdf/1704.02965.pdf.

[2] Anggiratih, Endang and Putra, Agfianto Eko. (2019). Ship Identification on Satellite Image Using Convolutional Neural Network and Random Forest, Indonesian Journal of Computing and Cybernetics Systems, 13:2, 1978-1520, DOI: 10.22146/ijccs.37461

[3] Castelluccio, M., Poggi, G, Sanson, C., Verdoliva, L. (2015). Land Use Classiﬁcation in Remote Sensing Images by Convolutional Neural Networks, arXiv:1508.00092v1.

[4] Guosheng Lin and Anton Milan and Chunhua Shen and Ian Reid. (2016). RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation, 1611.06612.

[5] Microsoft, https://github.com/Microsoft/USBuildingFootprints.

[6] Pelletier, C, Webb, Geoffrey, and Petitjean, F. (2019). Temporal Convolutional Neural Network for the Classiﬁcation of Satellite Image Time Series, Remote Sensing, 11(5):523, DOI: 10.3390/rs11050523.

[7] Seung-Wook Kim, Sung-Jin Cho, Kwang-Hyun Uhm, Seo-Won Ji, Sang-Won Lee, Sung-Jea Ko; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2019, pp. 0-0 

[8] Shermeyer, Jacob & Etten, Adam. (2018). The Effects of Super-Resolution on Object Detection Performance in Satellite Imagery.

[9] Yanfei Zhong, Feng Fei, Yanfei Liu, Bei Zhao, Hongzan Jiao & Liangpei Zhang (2017) SatCNN: satellite image dataset classification using agile convolutional neural networks, Remote Sensing Letters, 8:2, 136-145, DOI: 10.1080/2150704X.2016.1235299

