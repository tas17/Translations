from loader import load_data
import collections
import numpy as np
from helper import tokenize, pad, preprocess, sequence_to_text, logits_to_text
from models import simple_model
from sklearn.model_selection import train_test_split


english_sentences = load_data('data/small_vocab_en')
french_sentences = load_data('data/small_vocab_fr')
english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = \
    preprocess(english_sentences, french_sentences)

max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)
print('Shape of preproc_english_sentences', preproc_english_sentences.shape)
print('Shape of preproc_french_sentences', preproc_french_sentences.shape)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)

# Reshaping the input to work with a basic RNN
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)

X_train, X_test, y_train, y_test = train_test_split(tmp_x, preproc_french_sentences)

X_train = X_train.reshape((-1, y_train.shape[-2], 1))


# Train the neural network
simple_rnn_model = simple_model(
    X_train.shape,
    french_vocab_size)

print('fitting shapes', X_train.shape, "(", X_train.shape[:1], ")", french_vocab_size, preproc_french_sentences.shape)

print(simple_rnn_model.summary())

simple_rnn_model.fit(X_train, y_train, batch_size=1024, epochs=10, validation_split=0.2)

simple_rnn_model.save("models/GRUAlone")

# Print prediction(s)
print(X_train.shape)
print(X_train[0].shape)
def predict(i):
    print("Predicting i (" + str(i) + "):")
    print(logits_to_text(simple_rnn_model.predict(X_train[i].reshape((-1, y_train.shape[-2], 1)))[0], french_tokenizer))
    print(sequence_to_text(list(y_train[i].reshape(y_train[i].shape[0])),
                           french_tokenizer))
    print(sequence_to_text(list(X_train[i].reshape(X_train[i].shape[0])),
                           english_tokenizer))


predict(1)
predict(2)
predict(0)

