[Home/](https://rohanlone.github.io/Home/)Portfolio

### <em> Updated: 30/04/2023 </em>


# About

As a seasoned data scientist with a proven track record of success across various industries, I have extensive experience in developing and deploying end-to-end machine learning solutions. My skillset includes proficiency in Python, ML/DL modeling, NLP, data engineering, Apache Airflow, Microsoft Azure, AWS, and databases, among other tools and technologies.

I have a demonstrated ability to deliver scalable, reliable, and high-performing models and pipelines that are well-suited to production environments. Whether working with cloud platforms such as Microsoft Azure and AWS or leveraging cutting-edge machine learning algorithms, I am committed to developing and deploying data-driven solutions that create business value. 

---

# Skills 
Below are some of my skills, and I'm always looking to learn more.

| Skill              | Rating         | 
| -------------       |:-------------:| 
| Machine Learning     | 8/10 | 
| Deep Learning      | 8/10      |  
| Python | 9/10      |  
| Apache Airflow | 7/10      | 
| Docker        | 9/10      |  
| AWS         | 8/10      |  
| Microsoft Azure         | 8/10      | 
| C/C++     | 6/10      |  
| MYSQL | 9/10      |  





# Portfolio

Listed below are some of the projects that I have done in my own time.

---

## Project 1 
# [Water Quality Analysis & Data Modeling](https://www.kaggle.com/rohanlone4/water-quality-analysis-and-ensemble-modeling) 
Water Quality Analysis & Data Modeling is a project that focuses on assessing water quality and developing a predictive model for determining its suitability for human consumption. To assess whether a water sample is fit for human consumption, I developed a predictive model using an ensemble learning technique.

Here are the steps I took:

1. `DATA PREPROCESSING:` I cleaned the data by removing any null values and replacing them with the mean of the corresponding column.
2. `DATA VISUALIZATION:` I used the Seaborn library to visualize the data and identify any potential trends or patterns.
3. `DATA MODELING:` I trained a variety of machine learning models, including random forest, gradient boosting, AdaBoost, and LightGBM.
4. `MODEL EVALUATION:` I evaluated the performance of the models using a variety of metrics, including accuracy, precision, and recall.
5. `HYPERPARAMETER TUNING:` I tuned the hyperparameters of the random forest and LightGBM models to improve their accuracy.

The final model achieved an accuracy of 87%. This means that the model was able to correctly predict whether a water sample was fit for human consumption with a 87% accuracy. This model can be used to monitor water quality and identify potential problems. It can also be used to educate the public about the importance of water quality.

Here are some of the challenges I faced:
1. The data was noisy and imbalanced.
2. The models were computationally expensive to train and deploy.

Despite these challenges, I was able to develop a predictive model that can be used to assess water quality. This model has the potential to improve public health and protect the environment.

<img src="https://raw.githubusercontent.com/RohanLone/Home/gh-pages/assets/images/Water_Qulaity_Result.png?raw=true"/>


---

## Project 2 
# [Plant Leaf & Disease Recognition](https://www.kaggle.com/rohanlone4/plant-leaf-and-disease-recognition-using-fast-ai)
This project aims to develop a deep learning model to automatically identify plant leaf diseases. The model is trained on a dataset of images of healthy and diseased plant leaves. The model is developed using the `Fast.ai` as well as `Tensorflow` library, which is a popular deep learning libraries for Python.

Project Goals:
The goals of this project are to:

1. Develop a deep learning model that can accurately identify plant leaf diseases.
2. Deploy the model to a web application so that it can be used by farmers and other plant growers to identify and treat plant diseases.

The methodology for this project will involve the following steps:

1. `DATA COLLECTION:` The data used in this project collected from the Plant Village dataset, which is a publicly available dataset of images of healthy and diseased plant leaves.
2. `DATA PREPROCESSING:` The data is preprocessed to remove any noise or irrelevant features.
3. `MODEL DEVELOPMENT:` A deep learning model developed using the Fast.ai and Tensorflow library. The model will be trained on a subset of the data and evaluated on the remaining data. `Fast.ai:` Used the ResNet18 pre-trained model, which is a convolutional neural network (CNN) that has been trained on a large dataset of images. This allows you to start with a model that has already learned to recognize certain features in images, which can help you to train your model more quickly.
4. `MODEL DEPOYMENT:` The model will is deployed to a web application so that it can be used by farmers and other plant growers to identify and treat plant diseases.

Project Challenges:
1. The data was noisy and imbalanced.
2. The model did not able to generalize to new data.
3. The model was computationally expensive to train and deploy.

Project Benefits:
The benefits of this project include:

1. The ability to automatically identify plant leaf diseases.
2. The ability to reduce the spread of plant diseases.
3. The ability to improve crop yields.
4. The ability to save farmers money.
<img src="https://github.com/RohanLone/Home/blob/gh-pages/assets/images/Asset%204.png?raw=true"/>
&emsp;&emsp;&emsp; All the classes of plant disease dataset 

---

## Project 3
# [Soccer Player recognition using TensorFlow 2 Object Detection API](https://github.com/RohanLone/Tensorflow_Object_Detection_with_Tensorflow_2.0) 
This project aims to develop a model that can recognize soccer players in images and videos. The model was built using the TensorFlow 2 Object Detection API. The API provides a set of tools and pre-trained models that can be used to build object detection models.

The model was trained on a custom dataset that was created by me. The dataset consisted of images and videos of soccer players. The dataset was labeled using LabelImg, a free and open-source image annotation tool.

The model was deployed as a web application. The web application allows users to upload images and videos of soccer players and receive a prediction of whether the images contain soccer players.

Project Goals:

* Develop a model that can accurately recognize soccer players in images and videos.
* Deploy the model as a web application.
* Make the model available to the public.


The methodology for this project will involve the following steps:

1. `DATA COLLECTION:` In this project, I collected data from a soccer matches and the dataset was labeled using LabelImg, a free and open-source image annotation tool.
2. `DATA PREPROCESSING:` Converted the images to a format that the model could understand. Normalized the images so that they had a consistent size and brightness. Augmented the data by creating new images from the existing ones. This was done using a variety of techniques, such as rotating, flipping, and cropping the images.


3. `MODEL DEVELOPMENT:` The model was trained on `Google Colab`, a cloud-based platform that provides free access to GPUs for machine learning tasks. The model was stored on Google Drive, a cloud-based storage service. The pre-trained model `ssd_mobilenet_v2_320x320_coco17` was used to implement `transfer learning`.

4. `MODEL DEPOYMENT:` The model was deployed as a web application. The web application allows users to upload images and videos of soccer players and receive a prediction of whether the images contain soccer players.
<img src="https://github.com/RohanLone/Home/blob/gh-pages/assets/images/8.png?raw=true"/>


---


## Project 4
# [Disaster Tweets: NLP with Tensorflow](https://www.kaggle.com/rohanlone4/disaster-tweets-nlp-with-tensorflow)
Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

Predicting which Tweets are about real disasters and which ones are not.  

* Built a Keras model on Kaggle's Competition data: Natural Language Processing with Disaster Tweets. 
* Performed EDA on data and visualized with word cloud. 


<img src="https://github.com/RohanLone/Home/blob/gh-pages/assets/images/twitter.png?raw=true"/>

---


## Project  5
# [Heart Disease Prediction](https://github.com/RohanLone/Heart-Disease-Prediction) 
The data set contains continuous and categorical data from the UCI Machine Learning Repository. It is used to predict whether or not a patient has heart disease.
The dataset consists of 303 individuals data. There are 14 columns in the dataset, which are described below.
    
   1. age
   2. sex
   3. cp, chest pain
   4. restbp, resting blood pressure (in mm Hg)
   5. chol, serum cholesterol in mg/dl
   6. fbs, fasting blood sugar
   7. restecg, resting electrocardiographic results
   8. thalach, maximum heart rate achieved
   9. exang, exercise induced angina
   10. oldpeak, ST depression induced by exercise relative to rest
   11. slope, the slope of the peak exercise ST segment.
   12. ca, number of major vessels (0-3) colored by fluoroscopy
   13. thal, this is short of thalium heart scan.
   14. hd, diagnosis of heart disease, the predicted attribute

  *  Built an SVM classifier.
Before that dealt with the data, Prcoess like Data preprocessing, selection, identifying missing data, dealing with missing data is done.


<img src="https://github.com/RohanLone/Home/blob/gh-pages/assets/images/SVM.png?raw=true"/>

---


## Project  6

# [Word Emebedding](https://github.com/RohanLone/word_embedding) 
Word embedding is a concept used in `natural language processing` to describe the representation of words for text processing, which is usually in the form of a real-valued vector that embeds the meaning of the word and determines that words that are near in the vector space will have similar meanings.
* Built a Keras model on IMDB Review Dataset for finding the relationship between words.
* Used Embedding projector to visualize the data.


<img src="https://github.com/RohanLone/word_embedding/blob/main/Embedding%20Projector.png?raw=true"/>


---

## Project  7
# [Word Cloud](https://github.com/RohanLone/wordcloud) 
Word cloud is an image composed of words used in a particular text or subject, in which the size of each word indicates its frequency or importance.

Word clouds are a simple and cost-effective way to visualise text data.

* Created interactive web application using open source Python library - `Streamlit`

<img src="https://github.com/RohanLone/wordcloud/blob/main/Screenshots/Screenshot_Streamlit_app.png?raw=true"/>

---

## Project  8
# [Face Mask Detection](https://github.com/RohanLone/Face_mask_detection) 
* Created a model that detects human faces with mask and without mask from images.
* 50 Images are used for model building and labelled using LabelImg
* Used Google Colab (for model training), Google Drive (for storage). 
* Achieved 0.878 and 0.883 Average Precision and recall respectively.
* ssd_mobilenet_v2_320x320_coco17_tpu pretrained model used to implement transfer learning. 

<img src="https://github.com/RohanLone/Home/blob/gh-pages/assets/images/result.png?raw=true"/>

---

## Project  9
# [Face Verification](https://github.com/RohanLone/FaceVerification) 
* Built a deep learning model from scratch that verifies face.
* Used Tensorflow library to build model, This model learn information about object categories from one, or only a few, training samples/images.
* This model requires only one image of a person which can be used for face verification `(One Shot Learning)`.
* It can be used for image based attendance capturing system. 

---

## Project 10 

# [Object Detection using OpenCV](https://github.com/RohanLone/object_detection_opencv) 
For this example project created an `HSV` mask using opencv and python for identifying object purely by it's color
The trackbar can be used for change the color for detection of object in frame.

```Following output is for black color object```

<img src="https://github.com/RohanLone/object_detection_opencv/blob/main/Demo%20Videos/Demo.gif?raw=true"/>


---

## Project  11

# [Vehicle Number Plate Detection](https://github.com/RohanLone/Number-Plate-Detection-App) 
For this example project I built a number plate detector to identify number plate from frame. This could be useful for Motorway Road Tolling, vehicle theft prevention and parking management. They could take a picture or real time video of a vehicle and an app could serve them number plate. This is the underlying model for building something with those capabilities. 

This project strictly built on OpenCV. Created an application using Streamlit library. 
<img src="https://github.com/RohanLone/Number-Plate-Detection-App/blob/main/Demo/Demo.gif?raw=true"/>

---



##### © 2023 Rohan Lone
