# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:34:23 2020

@author: Quang
"""

import glob
import pickle
from os.path import basename

import numpy as np
from keras.applications.resnet_v2 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.sequence import pad_sequences


def load_extract_features_model(extract_model_path):
    model = load_model(extract_model_path)
    return model

def extract_features(filename, model):
    # load the models
    image = load_img(filename, target_size=(224,224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data
    image = np.expand_dims(image, axis=0)
    # preprocess image for model
    image = preprocess_input(image)
    # get features 
    feature = model.predict(image,verbose=0)
    return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length)    :
    # seed the generation process
    in_text = "startseq"
    
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]        
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict nextword        
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

def return_description(filename, extract_model, model, tokenizer, max_length):
    # extract feature
    photo = extract_features(filename, extract_model)
    
    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)
    
    # remove "startseq" and "endseq"
    description = description.replace("startseq ","").replace(" endseq","")
    
    return description

def main():
    extract_model_path = "./preprocess/ResNet50_feature_extraction.h5"
    tokenizer_path = "./preprocess/tokenizer.pkl"
    max_length = 34
    
    # load the tokenizer
    tokenizer = pickle.load(open(tokenizer_path,"rb"))
    print("[INFO] Loading tokenizer successfully...")
    
    # load extract feature mode
    extract_model = load_extract_features_model(extract_model_path)
    print("[INFO] Loading extracting feature model successfully...")
    
    # load image captioning model
    model = load_model('image_captioning_model.h5')
    print("[INFO] Loading captioning model successfully...")
    
    example_images = glob.glob("./examples/*.jpg")
    
    for filename in example_images:
        desription = return_description(filename,extract_model, model, tokenizer, max_length)
        print("%s: "%basename(filename), desription)
    
if __name__=="__main__":
    main()
