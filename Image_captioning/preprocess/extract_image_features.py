# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:27:05 2020

@author: Quang
"""

from os import listdir
from pickle import dump

import numpy as np
from keras.applications import ResNet50V2
from keras.applications.resnet_v2 import preprocess_input
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img



# extract features from each photo in the directory
def extract_features(directory):
    # load model
    model = ResNet50V2(weights="imagenet")
    model.layers.pop()
    # re-structure the model by replacing the last output layer
    model = Model(inputs=model.inputs, outputs = model.layers[-1].output)
    # sumarize
    print(model.summary())
      
    # save model 
    model.save("ResNet50_feature_extraction.h5")
    # extract features from each photo
    features = dict()
    num_images = len(listdir(directory))

    for index, name in enumerate(listdir(directory)):
        # load an image from file
        filename = directory + "/" + name
        image = load_img(filename, target_size=(224, 224))

        # convert image pixels to numpy array
        image = img_to_array(image)

        # reshape data
        image = np.expand_dims(image, axis=0)
        # image = image[np.newaxis,:]

        # preprocess image for model
        image = preprocess_input(image)

        # get features
        feature = model.predict(image, verbose=0)

        # get image_id
        image_id = name.split(".")[0]

        # store feature
        features[image_id] = feature
        print(">(%i/%i) %s" % (index, num_images, name))

    return features


def main():
    # extract features from all images
    directory = "../Flickr8k_Dataset"
    features = extract_features(directory)
    print("Extracted Features: %d" % len(features))

    # save to file
    dump(features, open("features.pkl", "wb"))


if __name__ == "__main__":
    main()
