import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator        # Generates batches of image tensor data with real time data augmentation
import numpy as np                                              # General purpose array processing package
from keras.preprocessing import image
from keras.models import load_model

model = load_model('D:/Machine Learning/AI Projects/Breast_Cancer_Classification/model/CNN.h5')

# dimensions of our images
img_width, img_height = 300, 300                                # Height and width of input image
batch_size = 2                      # No of samples to be passed to NN

test_data_dir = "D:/Machine Learning/AI Projects/Breast_Cancer_Classification/data/testing/test"

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)
TEST_SIZE = 10


probabilities = model.predict_generator(test_generator, TEST_SIZE)
for index, probability in enumerate(probabilities):
    image_path = test_data_dir + "/" + test_generator.filenames[index]
    img = mpimg.imread(image_path)
    plt.imshow(img)
    if probability > 0.5:
        plt.title("%.2f" % (probability[0]*100) + "% malignant")
    else:
        plt.title("%.2f" % ((1-probability[0])*100) + "% benign")
    plt.show()
