#Machine Learning
#Group-6
#MBTI Prediction
#This file contains CNN, RNN(commented) and LSTM(commented)
import os
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

# Download crawl-300d-2M embeddings from https://fasttext.cc/docs/en/crawl-vectors.html
CRAWL_EMBEDDING_DIR = 'C:/Users/satya/Documents/utasem2/ML'
CRAWL_EMBEDDING_FILE = 'crawl-300d-2M.vec'
CRAWL_EMBEDDING_PATH = os.path.join(CRAWL_EMBEDDING_DIR, CRAWL_EMBEDDING_FILE)

# Reading a CSV file with pandas
mbti_df = pd.read_csv('C:/Users/satya/Documents/utasem2/ML/mbti_1.csv')  # change path as per your system

# DATA PREPROCESSING

# Converting text in 'posts' column to lowercase
mbti_df["posts"] = mbti_df["posts"].str.lower()

# Removing URL links from posts
mbti_df["posts"] = mbti_df["posts"].str.replace(r'https?://[a-zA-Z0-9./-]*/[a-zA-Z0-9?=_.]*[_0-9.a-zA-Z/-]*', ' ')

# Removing numbers, non-alphanumeric characters, and extra whitespaces from posts
mbti_df["posts"] = mbti_df["posts"].apply(lambda x: re.sub(r'[0-9\W_+]', ' ', x))

# Removing multiple whitespaces from posts
mbti_df["posts"] = mbti_df["posts"].apply(lambda x: re.sub(r'\s+', ' ', x))

# Removing stopwords from posts
stop_words = set(stopwords.words("english"))
mbti_df["posts"] = mbti_df["posts"].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# TOKENIZATION

# Tokenization using word tokenization
tokenizer = Tokenizer(num_words=2500)
tokenizer.fit_on_texts(mbti_df['posts'])  # Fit tokenizer on the tokenized posts
sequences = tokenizer.texts_to_sequences(mbti_df['posts'])  # Convert tokenized posts to sequences of integers

# Load crawl-300d-2M embeddings
crawl_embeddings_index = {}
with open(CRAWL_EMBEDDING_PATH, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        crawl_embeddings_index[word] = coefs

# Create embedding matrix
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))
for word, i in tokenizer.word_index.items():
    embedding_vector = crawl_embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Reshaping input data for RNNs
max_sequence_length = max([len(seq) for seq in sequences])  # Find the maximum sequence length
X_padded = pad_sequences(sequences, maxlen=max_sequence_length)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the target variable
y_encoded = label_encoder.fit_transform(mbti_df['type'])

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

# Resampling to balance the dataset using Random Oversampling (only applied to training data)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Building the CNN model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 300, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
model.add(Conv1D(256, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(256, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(256, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the CNN model
model.fit(X_resampled, y_resampled, epochs=8, batch_size=128)

# Evaluate the model on the test data
loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Predict probabilities for each class
y_pred_prob = model.predict(X_test)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_prob, axis=1)

# Decode labels
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Get the names of all classes
class_names = label_encoder.classes_

# Calculates confusion matrix
conf_matrix = confusion_matrix(y_test_decoded, y_pred_decoded, labels=class_names)
print("Confusion Matrix:")
print(conf_matrix)



# Calculates metrics for the whole test dataset
test_f1_score = f1_score(y_test_decoded, y_pred_decoded, average='weighted')
test_precision = precision_score(y_test_decoded, y_pred_decoded, average='weighted')
test_recall = recall_score(y_test_decoded, y_pred_decoded, average='weighted')

print("\nTest F1 Score:", test_f1_score)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)




# # Define Simple RNN model
# simple_rnn_model = Sequential([
#     Embedding(len(tokenizer.word_index) + 1, 300, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False),
#     SimpleRNN(128),
#     Dense(len(label_encoder.classes_), activation='softmax')
# ])



# # Compile Simple RNN model
# simple_rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train Simple RNN model
# simple_rnn_model.fit(X_resampled, y_resampled, epochs=5, batch_size=128, validation_data=(X_test, y_test), verbose=1)

# # Evaluate Simple RNN model
# loss, simple_rnn_test_accuracy = simple_rnn_model.evaluate(X_test, y_test)
# print(f"Simple RNN Test Accuracy: {simple_rnn_test_accuracy}")

# # Define LSTM model
# lstm_model = Sequential([
#     Embedding(len(tokenizer.word_index) + 1, 300, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False),
#     LSTM(128),
#     Dense(len(label_encoder.classes_), activation='softmax')
# ])

# # Compile LSTM model
# lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train LSTM model
# lstm_model.fit(X_resampled, y_resampled, epochs=5, batch_size=128, validation_data=(X_test, y_test), verbose=1)

# # Evaluate LSTM model
# loss, lstm_test_accuracy = lstm_model.evaluate(X_test, y_test)
# print(f"LSTM Test Accuracy: {lstm_test_accuracy}")






