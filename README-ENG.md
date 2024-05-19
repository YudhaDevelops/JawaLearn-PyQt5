# APPLICATION OF AUGMENTED REALITY TECHNOLOGY USING CNN (CONVOLUTIONAL NEURAL NETWORK) IN LEARNING JAVANESE CHARACTERS

The JawaLearn application is an application built using machine learning, namely CNN (Convolutional Neural Network), and using image data as a learning tool for the machine. By training different machines it is hoped that it will help you understand the form of Javanese script.

# Purpose of Building the Application
1. Makes it easy for you to learn Javanese script
2. Creating an Augmented Reality application by combining AI technology which can make teaching and learning Javanese script easier.
3. Proving that Augmented Reality technology can be combined together using the Convolutional Neural Network (CNN) algorithm.
4. Contribute to overcoming students' difficulties in learning Javanese script using Augmented Reality technology

# JawaLearn Application Features
1. Classification of Javanese Script
   > This feature can be used to transliterate Javanese LEGENA characters with detection based on images of written characters entered either on the camera daci or the storage on your device.
   
3. AR Detection of Javanese Script
   > This feature can be used to transliterate Javanese LEGENA characters with real-time detection using the available camera. This feature gives meaning to the characters placed in front of the camera.

# Requirements Package For Run This Program
> [!NOTE]
> I am using python version ```3.9.19```. By using the python environment in miniconda3.

```
pip install keras==2.12.0
```
```
pip install numpy==1.23.5
```
```
pip install opencv_python==4.9.0.80
```
```
pip install opencv_python_headless==4.9.0.80
```
```
pip install Pillow==9.5.0
```
```
pip install Pillow==10.3.0
```
```
pip install psutil==5.9.8
```
```
pip install PyQt5==5.15.10
```
```
pip install PyQt5_sip==12.13.0
```
```
pip install tensorflow==2.12.0
```
```
pip install tensorflow_intel==2.12.0
```
```
pip install tflite_runtime==2.14.0
```

# If the program runs well, it will look like the image below
> [!IMPORTANT]
> You must create a model first with the model extensions being .tflite for AR features and .h5 for Javanese script classification features. or you can directly use my existing model, [Download here](https://github.com/YudhaDevelops/JawaLearn-PyQt5/releases/tag/Models-JawaLearn)

> [!NOTE]
> Or you can create your own model with your own database, using the program I use at kaggle,
> See [For Model Image Classification](https://www.kaggle.com/happyngoding/cnn-aksara-use-7-models-fix), and see [For Model AR / Javanese script detection object](https://www.kaggle.com/happyngoding/ssd-mobilenet-v2-python-3-10-12)

## 1. Running the Javanese Script Classification feature
![klasifikasi](https://github.com/YudhaDevelops/JawaLearn-PyQt5/assets/106727245/1c3d93c5-3727-441f-9d68-58131150f729)

## 2. Running the Javanese Script AR Detection feature
![objek_deteksi](https://github.com/YudhaDevelops/JawaLearn-PyQt5/assets/106727245/41b04ddc-e0f6-44e9-a9fd-94200fe83ab6)

# This is an AR program that uses a model as a brain or database that is used to transliterate from Javanese script to Latin script. 
The program will add a box around the Javanese script area and provide the meaning of the character at the top left of the detected Javanese script. Can be seen in the video below

https://github.com/YudhaDevelops/JawaLearn-PyQt5/assets/106727245/5e11c069-26fb-4b43-9a07-7d12ab41d471

