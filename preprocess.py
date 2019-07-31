'''
Preprocessing script for processing Sarcasm news data from 
https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home#Sarcasm_Headlines_Dataset_v2.json
Part of the Coursera course of NLP in Tensorflow week 1. 
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

def parse_data(file):
	for l in open(file,'r'):
		yield json.loads(l)
#with open("Sarcasm_Headlines_Dataset_v2.json", 'r') as f:
#	data = json.load(f)
data = parse_data("data/Sarcasm_Headlines_Dataset_v2.json")
headlines = []
labels = []
urls = []

for datum in data:
	headlines.append(datum['headline'])
	labels.append(datum['is_sarcastic'])
print('Number of headlines in the data : '+str(len(headlines)))

tokenizer = Tokenizer(oov_token = "<OOV>")
tokenizer.fit_on_texts(headlines)
word_index = tokenizer.word_index
#print(word_index)
sequences = tokenizer.texts_to_sequences(headlines)
padded = pad_sequences(sequences, padding="post")
print('Shape of padded matrix : '+str(padded.shape))