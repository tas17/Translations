from loader import load_data
from helper import tokenize, pad, preprocess, sequence_to_text, logits_to_text, text_to_sequence
from models import simple_model, simple_embed_model
import collections
import numpy as np
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

#1 X_train = X_train.reshape((-1, y_train.shape[-2], 1))
#1 X_test = X_test.reshape((-1, y_test.shape[-2], 1))
X_train = X_train.reshape((-1, y_train.shape[-2]))
X_test = X_test.reshape((-1, y_test.shape[-2]))


# Train the neural network
# model = simple_model(
#     X_train.shape,
#     french_vocab_size+1)

model = simple_embed_model(X_train.shape, english_vocab_size+1, french_vocab_size+1)

print('fitting shapes', X_train.shape, "(", X_train.shape[:1], ")", french_vocab_size, preproc_french_sentences.shape)

print(model.summary())

model.fit(X_train, y_train, batch_size=1024, epochs=50, validation_split=0.2)

model.save("models/GRUEmbedded")

# Print prediction(s)
print(X_train.shape)
print(X_train[0].shape)


def predict_verbose(i, X, Y):
    # !!!!!!!!! Only works for simple model ! 
    print("Predicting i (" + str(i) + "):")
    print('Original sentence')
    print(sequence_to_text(list(X[i].reshape(X[i].shape[0])),
                           english_tokenizer))
    print("Predicted translation ")
    print(logits_to_text(model.predict(X[i].reshape((-1, Y.shape[-2], 1)))[0], french_tokenizer))
    print('Correct Translation')
    print(sequence_to_text(list(Y[i].reshape(Y[i].shape[0])),
                           french_tokenizer))


def evaluate(k, X, Y):
    pred = text_to_sequence(logits_to_text(model.predict(X[k].reshape((-1, Y.shape[-2], 1)))[0], french_tokenizer), french_tokenizer)
    pred2 = logits_to_text(model.predict(X[k].reshape((-1, Y.shape[-2], 1)))[0], french_tokenizer)
    lis = np.array([1 if pred[i] == Y[k][i] else 0 for i, x in enumerate(pred)])
    return sum(lis) / lis.shape[0]


#1 predict_verbose(1, X_train, y_train)
#1 predict_verbose(2, X_train, y_train)
#1 predict_verbose(0, X_train, y_train)


print(model.metrics_names, model.evaluate(X_test, y_test))
