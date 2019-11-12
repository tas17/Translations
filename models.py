from keras.models import Sequential
from keras.losses import sparse_categorical_crossentropy
from keras.layers import GRU, LSTM, Dense, Dropout, Input, RepeatVector, CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.optimizers import Adam, TFOptimizer
from keras import Model
import keras.backend as K
import tensorflow as tf
import keras



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
                        embeddings_initializer=Constant(initializing_matrix), mask_zero=True))
    model.add(GRU(256, return_sequences=True))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(french_vocab_size, activation='softmax'))

    # Compile model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


latent_dim = 100
def encoder_decoderRMSProp(english_vocab_size, french_vocab_size):
    # Encoder
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(english_vocab_size, latent_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

    # Only keeping the states
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(french_vocab_size, latent_dim, mask_zero=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    # Use a softmax to generate a probability distribution over the target vocabulary for each time step
    decoder_dense = Dense(french_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    # Encode the input sequence to get the "thought vectors"
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    dec_emb2 = dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2)

    return model, encoder_model, decoder_model


def encoder_decoderAdam(english_vocab_size, french_vocab_size):
    # Encoder
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(english_vocab_size, latent_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

    # Only keeping the states
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(french_vocab_size, latent_dim, mask_zero=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    # Use a softmax to generate a probability distribution over the target vocabulary for each time step
    decoder_dense = Dense(french_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # Encode the input sequence to get the "thought vectors"
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    dec_emb2 = dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2)

    return model, encoder_model, 0


def categorical_accuracy_per_sequence(y_true, y_pred):
    return K.mean(K.min(K.equal(K.argmax(y_true, axis=-1),
        K.argmax(y_pred, axis=-1)), axis=-1))

from tensorflow.python.ops import math_ops

def custom_accuracySame(y_true, y_pred):
    return math_ops.cast(
                  math_ops.equal(
                                math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred, axis=-1)),
                        K.floatx())

def custom_accuracy(y_true, y_pred):
    return math_ops.reduce_min(
            math_ops.cast(
                math_ops.equal(
                    math_ops.argmax(y_true, axis=-1),
                    math_ops.argmax(y_pred, axis=-1)
                ), K.floatx() 
            ), axis=-1)

def encoder_decoderAdamBiggerLSTMCapacity(english_vocab_size, french_vocab_size):
    # Encoder
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(english_vocab_size, latent_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim*10, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

    # Only keeping the states
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(french_vocab_size, latent_dim*10, mask_zero=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim*10, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    # Use a softmax to generate a probability distribution over the target vocabulary for each time step
    decoder_dense = Dense(french_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # Encode the input sequence to get the "thought vectors"
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    dec_emb_layer = Embedding(french_vocab_size, latent_dim, mask_zero=True)
    dec_emb2 = dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2)

    return model, encoder_model, decoder_model


def encoder_decoderAdamBiggerEmbed(english_vocab_size, french_vocab_size):
    # Encoder
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(english_vocab_size, latent_dim*2, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim*2, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

    # Only keeping the states
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(french_vocab_size, latent_dim*2, mask_zero=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    # Use a softmax to generate a probability distribution over the target vocabulary for each time step
    decoder_dense = Dense(french_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # Encode the input sequence to get the "thought vectors"
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim*2,))
    decoder_state_input_c = Input(shape=(latent_dim*2,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    dec_emb2 = dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2)

    return model, encoder_model, decoder_model


def encoder_decoderAdamOneEmbed(english_vocab_size, french_vocab_size):
    # Encoder
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(english_vocab_size, latent_dim*2, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

    # Only keeping the states
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state
    decoder_inputs = Input(shape=(None,))
    # dec_emb_layer = Embedding(french_vocab_size, latent_dim*2, mask_zero=True)
    # dec_emb = dec_emb_layer(decoder_inputs)

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Use a softmax to generate a probability distribution over the target vocabulary for each time step
    decoder_dense = Dense(french_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # Encode the input sequence to get the "thought vectors"
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # dec_emb2 = dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2)

    return model, encoder_model, decoder_model


def encoder_decoderAdamFast(english_vocab_size, french_vocab_size):
    # Encoder
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(english_vocab_size, latent_dim)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

    # Only keeping the states
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(french_vocab_size, latent_dim, mask_zero=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    # Use a softmax to generate a probability distribution over the target vocabulary for each time step
    decoder_dense = Dense(french_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # Encode the input sequence to get the "thought vectors"
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    dec_emb2 = dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2)

    return model, encoder_model, decoder_model

