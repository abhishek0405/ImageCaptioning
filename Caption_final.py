#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir
from pickle import dump,load #to save objects on disc
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img #take in pil image
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input #preprocess to extract features
from keras.models import Model #to include layers needed.
from keras.preprocessing.sequence import pad_sequences
#import keras
from numpy import argmax
from keras.models import load_model


# In[2]:


model = load_model('new_model_4.h5')



# In[3]:


def extract_features_test(filename):
# load the model
    model = VGG16()
    
# re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    
# load the photo
    image = load_img(filename, target_size=(224, 224))
# convert the image pixels to a numpy array
    image = img_to_array(image)
# reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
# get features
    feature = model.predict(image, verbose=0)
    return feature


# In[5]:


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# In[4]:


def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
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


# In[6]:


tokenizer=load(open('tokenizer.pkl','rb'))


# In[7]:


def predict_caption(photo):
    feat=extract_features_test(photo)
    caption1=generate_desc(model,tokenizer,feat,34)
    return ' '.join(caption1.split()[1:-1])


# In[8]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:



    


# In[ ]:





# In[ ]:




