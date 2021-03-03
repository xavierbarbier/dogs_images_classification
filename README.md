# Dog breed prediction API

## Configuration
Environment : Google Colab with GPU
Libraries used : TensorFlow, Keras, Pandas, Numpy, Matplotlib


## Data
**Stanford Dogs Dataset : ** http://vision.stanford.edu/aditya86/ImageNetDogs/.

The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization.

## Preprocessing
* Rescaling
* image augmentation

## CNN (from scratch)
Differents architectures were tested. Image augmentation and dropout layer were used to asses overfitting.

Notebook => https://github.com/xavierbarbier/dogs_images_classification/blob/main/dogs_class_CNN.ipynb

## Transfert learning
MobilNet V3 was used. Image augmentation, batch normalisation, adding dense and/or dropout layers were used to asses overfitting.

Notebook => https://github.com/xavierbarbier/dogs_images_classification/blob/main/dogs_class_transfert_learning.ipynb

## Fine tunning
Keras tuner was used to search for best learning rate.

Notebook => https://github.com/xavierbarbier/dogs_images_classification/blob/main/dogs_class_keras_tuner.ipynb

## API
App was deploy on Heroku using Dash library.

API => https://dogs-prediction.herokuapp.com/

## Project slides (in French)
Slides => https://github.com/xavierbarbier/dogs_images_classification/blob/main/dogs_class_CNN_presentation.pdf
