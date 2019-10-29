from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.losses import sparse_categorical_crossentropy


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
