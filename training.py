import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import random

# === Load data ===
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data.json', encoding="utf-8").read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# === Preprocessing ===
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# simpan words & classes
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# === Tokenizer untuk urutan kata ===
tokenizer = Tokenizer(num_words=5000, lower=True, oov_token="<OOV>")
all_patterns = [" ".join([lemmatizer.lemmatize(w.lower()) for w in doc[0]]) for doc in documents]
tokenizer.fit_on_texts(all_patterns)

# simpan tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# === Convert patterns jadi sequence ===
sequences = tokenizer.texts_to_sequences(all_patterns)
max_len = max(len(seq) for seq in sequences)  # panjang max input
X = pad_sequences(sequences, maxlen=max_len, padding='post')

# === One-hot encode labels ===
output_empty = [0] * len(classes)
y = []
for doc in documents:
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    y.append(output_row)

X = np.array(X)
y = np.array(y)

print("Training data created")
print("Shape X:", X.shape, "Shape Y:", y.shape)

# === Build LSTM model ===
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compile
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train
hist = model.fit(X, y, epochs=200, batch_size=5, verbose=1)
model.save('model.h5')

print("Model created and saved with LSTM")