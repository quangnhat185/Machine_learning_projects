# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:10:38 2020

@author: Quang
"""
import os.path
from pickle import dump, load

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, Embedding, Input
from keras.layers.merge import add
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from preprocess.generate_tokenizer import (create_tokenizer,
                                           load_clean_descriptions, load_set,
                                           to_lines)
from preprocess.preprocess_text_data import load_doc


# load photo features
def load_photo_features(filename,dataset):
    # load all features
    all_features = load(open(filename,'rb'))
    # Keep only features from image existing in dataset
    features = {k: all_features[k] for k in dataset}
    return features

# calculate the length of the description with the most words
def max_length_descriptions(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

# create sequence of images, input sequences and output words of an image
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1,X2,y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into mutiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pairs
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sentence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
            
    return np.array(X1), np.array(X2), np.array(y)

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    # loop for every image
    while True:
        for key, desc_list in descriptions.items():
            # retrieve photo feature
            photo = photos[key][0]
            in_img, in_seq, out_world = create_sequences(tokenizer, 
                                                         max_length, 
                                                         desc_list, 
                                                         photo, 
                                                         vocab_size)
            yield [[in_img, in_seq], out_world]
    

# define the captioning model
def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero = True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256,activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # combine them together
    model = Model(inputs=[inputs1, inputs2], outputs = outputs)
    model.compile(loss="categorical_crossentropy", optimizer='adam')
    
    # sumarize model
    print(model.summary())
    return model

# save model architectur as json file
def save_model_json(model):
    model_json = model.to_json()
    with open("image_captioning_LSTM_RetNet50.json", "w") as json_file:
      json_file.write(model_json)
      
def plot_training_history(result):
    plt.figure(figsize=(8,5))    
    plt.plot(result.history['loss'], label='training loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.savefig("Training_result.jpg",dpi=300)
        

    
def main():
    # load training dataset
    filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
    train = load_set(filename)
    print("Training dataset: %d images" %len(train))
    
    # load descriptions
    train_descriptions = load_clean_descriptions("./preprocess/descriptions.txt",train)
    print("Training descriptions: %d"%len(train_descriptions))

    # photo features
    train_features  = load_photo_features("./preprocess/features.pkl",train)
    print("Training photos: %d"%len(train_features))
    
    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print("Training vocabulary size: %d"%vocab_size)
    
    # determiner the maximum sequence length
    max_length = max_length_descriptions(train_descriptions)
    print("Training description length: %d"%max_length)
    
#    # load test set
#    filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
#    test = load_set(filename)
#    print('Dataset: %d' % len(test))
#    # descriptions
#    test_descriptions = load_clean_descriptions('./preprocess/descriptions.txt', test)
#    print('Descriptions: test=%d' % len(test_descriptions))
#    # photo features
#    test_features = load_photo_features('./preprocess/features.pkl', test)
#    print('Photos: test=%d' % len(test_features))
  
    # initialize model
    model = define_model(vocab_size, max_length)
        
    # define checkpoint callbacks
    filepath = "/content/drive/My Drive/Sharing_storage1/weights/image_captioning_model.h5"
    checkpoint = ModelCheckpoint(filepath, 
                                 monitor="loss", 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode="min")
    
    if os.path.isfile(os.path.basename(filepath)):
        model.load_weights(os.path.basename(filepath))
        print("Loading weight successfully...")
        
    
    epochs = 30
    steps = len(train_descriptions)
    
#    generator = data_generator(train_descriptions,
#                               train_features,
#                               tokenizer,
#                               max_length,
#                               vocab_size)
    
    result = model.fit_generator(data_generator(train_descriptions,train_features, tokenizer, max_length, vocab_size), 
                                 epochs=20, 
                                 steps_per_epoch=steps, 
                                 verbose=1, 
                                 callbacks=[checkpoint])

    plot_training_history(result)
    
if __name__=="__main__":
    main()    
