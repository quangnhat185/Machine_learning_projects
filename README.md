## Machine_learning_projects
This repository contains my first personal Machine Learning Projects. To view my notebook directly from github, you can use [nbviewer](https://nbviewer.jupyter.org/). Belows are brief descriptions and images about my projects:

1. [Dog and cat classification (95% accuracy)](#dog-and-cat-classification)
2. [Fashion classification with fashion_mnnist datasets from Keras (91% accuracy)](#Fashion-classification-with-fashion_mnnist-datasets-from-Keras)
3. [Traffic signs classification with German Traffic Sign dataset on Kaggle (97% accuracy)](#Traffic-signs-classification-with-German-Traffic-Sign-dataset-on-Kaggle)
4. [Object classisfication with YOLOv3](#Object-classification-with-YOLOv3)
5. [Gun detection with YOLOv3](#Gun-detection-with-YOLOv3)
6. [Car brand classification with EfficientNet (70% accuracy)](#Car-brand-classification-with-EfficientNet)

#### [Dog and Cat classification][1]:
  - [Dataset](http://bit.ly/30k1jgs): this datset contains 25,000 images of dog and cat collected from Internet.
  - My customized model contains 12 layers had achieved accuracy 87.8%. Used transfer learning with pre-trained Resnet50, I have improved my model accuracy up to nearly 98%.

<p align="center">
  <img src="./Dog_Cat_classification/Dog_cat_prediction.png">
</p>
<br>

#### [Fashion classification with fashion_mnnist datasets from Keras][2]:
  - Dataset: The fashion_mnist datasets of Keras contains of 60,000 examples and a test set of 10,000 examples
  - My model consists of 597,786 parameters and was trained for 10 epochs. The accuracy is nearly 95%
<p align="center">
  <img src="./Fashion_classification/F1.jpg">
</p>
<br>

#### [Traffic signs classification with German Traffic Sign dataset on Kaggle][3]:
  - Dataset: The Kaggle German Traffic Sign dataset consists of 40 classes and more than 50,000 images in total.
  - The mode created in this project contains 478,763 parameter and was trained for 20 epochs. The final accuracy after testing with new test data is 96.8%
<p align="center">
  <img src="./Traffic_signs_classification/Traffic_sign_test.png">
</p>
<br>

#### [Object classification with YOLOv3][4]:
  - Dataset: COCO dataset is a large-scale object-detection dataset which contains of approximately 330K images.
  - In this project, I applied YOLOv3 algorithm on the dataset of COCO and pass an image and video through. The results can be seen as below figures:

<p align="center">
  <img src="./Object_classification_Yolov3/T1.png">
  <img src="./Object_classification_Yolov3/Vietnam_traffic.gif", width = "800">
</p>
<br>

#### [Gun detection with YOLOv3][5]:
  - [Dataset](http://www.mediafire.com/file/pvfircmboaelkxc/Gun_data_labeled.zip/file): 3k images of hand gun and their label annotation (bounding box coordination)
  - Model was trained on Colab with 900 epochs using Yolov3.

<p align="center">
  <img src="./Gun_detection_Yolov3/gun_detection.png">
  <img src="./Gun_detection_Yolov3/gun_detection.gif", width = "600">
</p>
<br>

#### [Car brand classification with EfficientNet][6]
- [Dataset](https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder): Stanford car dataset which contains of 16k images from 196 classes of car brands
- Model was trained with fine-tune pre-trained EfficientNet and acquire the accuracy of 73%.

<p align="center">
  <img src="./Car_brand_classification/prediction_result.png">
</p>
<br>

#### [License_Plate_Detection_and_Recognition][7]:
- In this project, I created a pipeline that detect and read car license plate in Europe. The piple contains three main statges:
  1. Detect lincense plate from a rear car image using WPOD model.
  1. Grab contour area of each digit using OpenCV
  1. Predict the extracted contour area with SVM model.

  <p align="center">
  <img src="./License_Plate_Detection_and_Recognition/final_result.jpg", width= 700>
  </p>

[1]:/Dog_Cat_classification
[2]:/Fashion_classification
[3]:/Traffic_signs_classification
[4]:/Traffic_classification_Yolov3
[5]:/Gun_detection_Yolov3
[6]:/Car_brand_classification
[7]:/License_Plate_Detection_and_Recognition
