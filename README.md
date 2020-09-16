# Breast_Cancer_Classification
Binary classification between malignant and benign breast cancer image dataset
The dataset for this project can be downloaded from this page: https://www.kaggle.com/ankur1809/breakhist-dataset
The dataset contains Breast Cancer Histopathological Images captured at different manginfication factors(40x, 100x, 200x, 400x).
I have clubbed all the images with the different magnification factors and then sorted them into 2 primary folders namely "malignant" and "benign".
I split up the dataset further by sorting them into "train", "validate" and "testing" folders.

The "train" folder contains a total of 6141 images of which 4195 images are of "malignant" breast cancer and 1946 images are of "benign" breast cancer.
Similarily, the "validate" folder contains 1758 total images for validation of which 1229 images are of "malignant" breast cancer and 529 images are of "benign" breast cancer.
I have kept 5 images of "malignant" and "benign" breast cancer each for taking the predictions.
All the images are of different dimensions but have been resized to the dimension of 300x300 for training the classifier.
The image classifier has been trained for a total of 20 epochs with a batch size of 20.

The model weights have been saved in a file with a .h5 extension, I have not uploaded the file on Github due to size restrictions.


Cheers!!!
