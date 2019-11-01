from keras.models import Sequential
from keras.losses import sparse_categorical_crossentropy
from keras.layers import GRU, Dense, Dropout, Masking
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.optimizers import Adam, TFOptimizer
import tensorflow as tf



def simple_model(input_shape, final_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param final_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """

    model = Sequential()
    model.add(GRU(256, input_shape=input_shape[1:], return_sequences=True))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(final_size, activation='softmax'))

    # Compile model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def simple_embed_model(input_shape, english_vocab_size, french_vocab_size):
    """
    Build and train a RNN model using word embedding on x and y
    :param input_shape: Tuple of input shape
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """

    print(input_shape, input_shape[1], input_shape[1:])
    model = Sequential()
    model.add(Embedding(english_vocab_size, 256, input_length=input_shape[1], input_shape=input_shape[1:]))
    model.add(GRU(256, return_sequences=True))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(french_vocab_size, activation='softmax'))

    # Compile model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(0.005),
                  metrics=['accuracy'])
    return model


def initialized_embed_model(input_shape, english_vocab_size, french_vocab_size, initializing_matrix):
    model = Sequential()
    model.add(Embedding(english_vocab_size, 256, input_length=input_shape[1], input_shape=input_shape[1:],
                        embeddings_initializer=Constant(initializing_matrix)))
    model.add(GRU(256, return_sequences=True))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(french_vocab_size, activation='softmax'))

    # Compile model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def padAndMask(input_shape, english_vocab_size, french_vocab_size, initializing_matrix):
    print(input_shape, input_shape[1], input_shape[1:])
    model = Sequential()
    # model.add(Masking(mask_value=0, input_shape=(None, input_shape[1])))
    model.add(Embedding(english_vocab_size, 256, input_length=input_shape[1], input_shape=input_shape[1:],
                        embeddings_initializer=Constant(initializing_matrix)))
    model.add(GRU(256, return_sequences=True))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(french_vocab_size, activation='softmax'))

    # Compile model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


