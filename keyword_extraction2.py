# -*- coding: utf-8 -*-

#classic approach based on frequency words

import re
from collections import Counter
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt


# Download the 'punkt' resource
nltk.download('punkt')

# Download the 'stopwords' resource
nltk.download('stopwords') # Download the stopwords resource

#sample text
text = """ Will the Next iPhone Have a 3D Camera?
Apple has filed for a patent that would involve taking multiple pictures and meshing them together to create a 3-dimensional image — a process that would include the use of two cameras instead of one — according to AppleInsider. It remains to be seen as to whether Apple will jump on the 3D bandwagon, but many companies have already embraced the technology. There are actually a number of phones on the market that already use 3D cameras and displays. The EVO 3D, which debuted at CTIA Wireless 2011, uses a pair of cameras on the back of the phone to capture high-definition 3D video. The patent indicates that the technology would allow the iPhone to shoot still images in 3D and also record high-definition 3D video, according to the report. Both the EVO 3D and Nintendo’s latest handheld gaming device, the 3DS, uses a technology called a parallax barrier — which basically only lets each eye see a certain set of pixels ont he screen. That means that each eye sees something different, and the brain combines the images. The net effect is an illusion of depth and a 3D image without needing cumbersome 3D glasses. The patent doesn’t indicate whether Apple’s next iteration of the iPhone would carry a similar screen using parallax barrier technology. But 3D has not emerged as the dominant new trend in display technology, as there are some complaints about whether it places undue strain on eyes. The Nintendo 3DS, for example, warns users to take a break from time to time so they do not strain their eyes. Apple is expected to delay the release of its next iPhone until the fall. That would give the company more than enough time to introduce a number of new pieces of technology to the device, including access to the next generation of wireless networks. Tags: 3D, 3DS, Apple iPhone, EVO 3D, iPhone, Nintendo 3DS Companies: Apple, nintendo Copyright 2011 VentureBeat. All Rights Reserved. VentureBeat is an independent technology blog. Read More ""
"""

# PREPROCESSING
#normalization
text = text.lower()

# remove special characters
text = re.sub(r'\W+', ' ', text)

#tokenize
tokens = word_tokenize(text)

# remove stopwords
other_stopwords = {"make", "example", "fortunately", "said"}
stop_words = set(stopwords.words('english')).union(other_stopwords)
filtered_tokens = [word for word in tokens if word not in stop_words]


#Calculate frequencies
word_freq = Counter(filtered_tokens)
print("Word Frequencies:", word_freq)

# keywords = words with frequencyy > 2
frequent_words = {word: count for word, count in word_freq.items() if count > 2}
print("Key words", frequent_words)

print("Keywords", len(frequent_words))
#Top 10 key words
#top_words = word_freq.most_common(10)
#words, counts = zip(*top_words)
#print("Top Words:", top_words)

# Plot

plt.bar(frequent_words.keys(), frequent_words.values(), color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

## Code based on https://github.com/boudinfl/pke?tab=readme-ov-file

!pip install git+https://github.com/boudinfl/pke.git
# download the english model
!python -m spacy download en_core_web_sm

import pke

# initialize a TopicRank keyphrase extraction model
extractor = pke.unsupervised.TopicRank()

#sample text
text = """ Will the Next iPhone Have a 3D Camera?
Apple has filed for a patent that would involve taking multiple pictures and meshing them together to create a 3-dimensional image — a process that would include the use of two cameras instead of one — according to AppleInsider. It remains to be seen as to whether Apple will jump on the 3D bandwagon, but many companies have already embraced the technology. There are actually a number of phones on the market that already use 3D cameras and displays. The EVO 3D, which debuted at CTIA Wireless 2011, uses a pair of cameras on the back of the phone to capture high-definition 3D video. The patent indicates that the technology would allow the iPhone to shoot still images in 3D and also record high-definition 3D video, according to the report. Both the EVO 3D and Nintendo’s latest handheld gaming device, the 3DS, uses a technology called a parallax barrier — which basically only lets each eye see a certain set of pixels ont he screen. That means that each eye sees something different, and the brain combines the images. The net effect is an illusion of depth and a 3D image without needing cumbersome 3D glasses. The patent doesn’t indicate whether Apple’s next iteration of the iPhone would carry a similar screen using parallax barrier technology. But 3D has not emerged as the dominant new trend in display technology, as there are some complaints about whether it places undue strain on eyes. The Nintendo 3DS, for example, warns users to take a break from time to time so they do not strain their eyes. Apple is expected to delay the release of its next iPhone until the fall. That would give the company more than enough time to introduce a number of new pieces of technology to the device, including access to the next generation of wireless networks. Tags: 3D, 3DS, Apple iPhone, EVO 3D, iPhone, Nintendo 3DS Companies: Apple, nintendo Copyright 2011 VentureBeat. All Rights Reserved. VentureBeat is an independent technology blog. Read More ""
"""

# load the document using the initialized model
# text preprocessing is carried out using spacy
extractor.load_document(input=text, language='en')

# identify the keyphrase candidates using TopicRank's default strategy
# i.e. the longest sequences of nouns and adjectives `(Noun|Adj)*`
extractor.candidate_selection()

# In TopicRank, candidate weighting is a three-step process:
#  1. candidate clustering (grouping keyphrase candidates into topics)
#  2. graph construction (building a complete-weighted-graph of topics)
#  3. rank topics (nodes) using a random walk algorithm
extractor.candidate_weighting()

# Get the N-best candidates (here, 10) as keyphrases
keyphrases = extractor.get_n_best(n=20, stemming=False)

# for each of the best candidates
for i, (candidate, score) in enumerate(keyphrases):

    # print out the its rank, phrase and score
    print("rank {}: {} ({})".format(i, candidate, score))

# Commented out IPython magic to ensure Python compatibility.
import networkx as nx
import matplotlib.pyplot as plt
# %matplotlib inline

# set the labels as list of candidates for each topic
labels = {i: ';'.join(topic) for i, topic in enumerate(extractor.topics)}

# set the weights of the edges
edge_weights = [extractor.graph[u][v]['weight'] for u,v in extractor.graph.edges()]

# set the weights of the nodes (topic weights are stored in _w attribute)
sizes = [10e3*extractor._w[i] for i, topic in enumerate(extractor.topics)]

# draw the graph
nx.draw_shell(extractor.graph, with_labels=True, labels=labels, width=edge_weights, node_size=sizes)

from google.colab import drive
drive.mount("/content/drive")

import pke
import os
folder_path = "/content/drive/MyDrive/Colab Notebooks/FPA Applied Math/Texts"

files = sorted(os.listdir(folder_path))
keyphrases_texts = []
for file_name in files:
  extractor = pke.unsupervised.TopicRank()
  keyphrases_text = []
  file_path = os.path.join(folder_path, file_name)
  # Verificar si es un archivo (no una carpeta)
  if os.path.isfile(file_path):
      with open(file_path, 'r', encoding='utf-8') as file:
          content = file.read()
          #print(f"Contenido de {file_name}:")
          #print(content)

          extractor.load_document(input=content, language='en')
          extractor.candidate_selection()
          extractor.candidate_weighting()
          keyphrases = extractor.get_n_best(n=50, stemming=False)
          #print(keyphrases)
          for i, (candidate, score) in enumerate(keyphrases):
            keyphrases_text.append(candidate)
          #for i, (candidate, score) in enumerate(keyphrases):
            # print out the its rank, phrase and score
            #print("rank {}: {} ({})".format(i, candidate, score))
      keyphrases_texts.append(keyphrases_text)
#print(keyphrases_texts)

folder_path = "/content/drive/MyDrive/Colab Notebooks/FPA Applied Math/keys"

files = sorted(os.listdir(folder_path))
keys_total = []
for file_name in files:
  file_path = os.path.join(folder_path, file_name)
  # Verificar si es un archivo (no una carpeta)
  if os.path.isfile(file_path):
      with open(file_path, 'r', encoding='utf-8') as file:
          #content = file.read()
          #print(f"Contenido de {file_name}:")
          keys = [line.strip() for line in file]
          #print(keys)
      keys_total.append(keys)
#print(keys_total)

for i in range(len(keyphrases_texts)):
  TP = len(list(set(keyphrases_texts[i]) & set(keys_total[i])))
  pos_total = len(keyphrases_texts[i])
  corr_total = len(keys_total[i])
  precision = float(TP)/pos_total
  recall = float(TP)/corr_total
  f1 = 2*((precision*recall)/(precision+recall)) if TP != 0 else 0
  print(f"text {i+1}, TP: {TP}, Total positivos: {pos_total}, Total correctos:{corr_total}, recall:{recall:.2f}, precision: {precision:.2f}, f1: {f1:.2f}")

# RNN approach
import os
import zipfile
import urllib.request
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import Sequence

# dataset
url = "https://github.com/LIAAD/KeywordExtractor-Datasets/blob/master/datasets/500N-KPCrowd-v1.1.zip?raw=true"
zip_file = "500N-KPCrowd-v1.1.zip"
extracted_folder_path = "500N-KPCrowd-v1.1/500N-KPCrowd-v1.1"
urllib.request.urlretrieve(url, zip_file)
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

docs_folder = os.path.join(extracted_folder_path, "docsutf8")
keys_folder = os.path.join(extracted_folder_path, "keys")

txt_files = sorted([f for f in os.listdir(docs_folder) if f.endswith('.txt')])
key_files = sorted([f for f in os.listdir(keys_folder) if f.endswith('.key')])

texts = []
keywords = []
for txt_file, key_file in zip(txt_files, key_files):
    with open(os.path.join(docs_folder, txt_file), 'r', encoding='utf-8') as file:
        article_text = file.read()
        texts.append(article_text)

    with open(os.path.join(keys_folder, key_file), 'r', encoding='utf-8') as file:
        article_keywords = file.read().strip().split("\n")
        keywords.append(article_keywords)

# tokenize and pad text
text_tokenizer = Tokenizer()
text_tokenizer.fit_on_texts(texts)
X_sequences = text_tokenizer.texts_to_sequences(texts)
max_len = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, padding='post', maxlen=max_len)

keyword_tokenizer = Tokenizer()
keyword_tokenizer.fit_on_texts([kw for kws in keywords for kw in kws])

# keywords to indices and pad
y_sequences = [[keyword_tokenizer.word_index.get(kw, 0) for kw in kws] for kws in keywords]
y_padded = pad_sequences(y_sequences, padding='post', maxlen=max_len)

# getting our test and train sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, test_size=0.2, random_state=42)

# data generator to be able to train using batches to avoid running out of mem :)
class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size=16):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        return X_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# actual RNN model starts here
model = Sequential()
model.add(Embedding(input_dim=len(text_tokenizer.word_index) + 1, output_dim=100, input_length=max_len))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(Dense(len(keyword_tokenizer.word_index) + 1, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 8
train_generator = DataGenerator(X_train, y_train, batch_size=batch_size)
test_generator = DataGenerator(X_test, y_test, batch_size=batch_size)

# training rnn
model.fit(train_generator, epochs=5, validation_data=test_generator)

# evaluating rnn
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

from tensorflow.keras.utils import plot_model
model.build()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# examples:
# get index of specific files:
test = 'fashion-20902824'
#test = 'science-20871631'
test_index = 0
for index, file in enumerate(txt_files):
    if file == test:
        test_index = index
        break

example_text = texts[test_index]
example_seq = text_tokenizer.texts_to_sequences([example_text])
example_seq_padded = pad_sequences(example_seq, maxlen=max_len, padding='post')
predicted = model.predict(example_seq_padded)
print(predicted)
predicted_keywords_indices = predicted.argmax(axis=-1)
predicted_keywords = [keyword_tokenizer.index_word.get(i, '') for i in predicted_keywords_indices[test_index] if i != 0]

print(f"Original Keywords: {keywords[test_index]}")
print(f"Predicted Keywords: {predicted_keywords}")

# actual RNN model starts here
model = Sequential()
model.add(Embedding(input_dim=len(text_tokenizer.word_index) + 1, output_dim=100, input_length=max_len))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(Dense(len(keyword_tokenizer.word_index) + 1, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 16
train_generator = DataGenerator(X_train, y_train, batch_size=batch_size)
test_generator = DataGenerator(X_test, y_test, batch_size=batch_size)

# training rnn
model.fit(train_generator, epochs=5, validation_data=test_generator)

# evaluating rnn
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
