from loader import load_data
from helper import tokenize, pad, preprocess, sequence_to_text, logits_to_text, text_to_sequence
from models import encoder_decoderRMSProp, encoder_decoderAdam, encoder_decoderAdamBiggerEmbed, \
    encoder_decoderAdamOneEmbed, encoder_decoderAdamFast, encoder_decoderAdamBiggerLSTMCapacity
import collections
import numpy as np
import gensim
import re
from sklearn.model_selection import train_test_split


english_sentences = []
french_sentences = []

# data_path = 'fra-eng/fra.txt'
# with open(data_path, 'r', encoding='utf-8') as f:
#     lines = f.read().split('\n')
#
# for line in lines:
#     a = line.split('\t')
#     if len(a) >= 2:
#         input_text = a[0].lower()
#         target_text = a[1].lower()
#         input_text = re.sub(r"[.,?!'-;<>#{()}^]", r"", input_text)
#         target_text = re.sub(r"[.,?!'-;<>#{()}^]", r"", target_text)
#         input_text = re.sub(r'"', r"", input_text)
#         target_text = re.sub(r'"', r"", target_text)
#         target_text = "<start> " + target_text + " <end>"
#         input_text = re.sub(" +", r" ", input_text)
#         target_text = re.sub(" +", r" ", target_text)
#         english_sentences.append(input_text)
#         french_sentences.append(target_text)

f1 = 'training/news-commentary-v9.fr-en.fr'
e1 = 'training/news-commentary-v9.fr-en.en'
f2 = 'un/undoc.2000.fr-en.fr'
e2 = 'un/undoc.2000.fr-en.en'
f3 = 'europarl-v7.fr-en.fr'
e3 = 'europarl-v7.fr-en.en'

b = True
with open(e1, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines:
    input_text = line
    input_text = re.sub(r"[.,?!'-;<>#{()}^]", r"", input_text)
    input_text = re.sub(r'"', r"", input_text)
    input_text = re.sub(" +", r" ", input_text)
    english_sentences.append(input_text)
    if b:
        b = False
        print('First line of text 1')
        print(input_text)

b = True
with open(e2, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines:
    input_text = line
    input_text = re.sub(r"[.,?!'-;<>#{()}^]", r"", input_text)
    input_text = re.sub(r'"', r"", input_text)
    input_text = re.sub(" +", r" ", input_text)
    english_sentences.append(input_text)
    if b:
        b = False
        print('First line of text 2')
        print(input_text)

b = True
with open(e3, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines:
    input_text = line
    input_text = re.sub(r"[.,?!'-;<>#{()}^]", r"", input_text)
    input_text = re.sub(r'"', r"", input_text)
    input_text = re.sub(" +", r" ", input_text)
    english_sentences.append(input_text)
    if b:
        b = False
        print('First line of text 3')
        print(input_text)

b = True
with open(f1, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines:
    target_text = line
    target_text = re.sub(r"[.,?!'-;<>#{()}^]", r"", target_text)
    target_text = re.sub(r'"', r"", target_text)
    target_text = "<start> " + target_text + " <end>"
    target_text = re.sub(" +", r" ", target_text)
    french_sentences.append(target_text)
    if b:
        b = False
        print('First line of text 1')
        print(target_text)

b = True
with open(f2, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines:
    target_text = line
    target_text = re.sub(r"[.,?!'-;<>#{()}^]", r"", target_text)
    target_text = re.sub(r'"', r"", target_text)
    target_text = "<start> " + target_text + " <end>"
    target_text = re.sub(" +", r" ", target_text)
    french_sentences.append(target_text)
    if b:
        b = False
        print('First line of text 2')
        print(target_text)

b = True
with open(f3, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines:
    target_text = line
    target_text = re.sub(r"[.,?!'-;<>#{()}^]", r"", target_text)
    target_text = re.sub(r'"', r"", target_text)
    target_text = "<start> " + target_text + " <end>"
    target_text = re.sub(" +", r" ", target_text)
    french_sentences.append(target_text)
    if b:
        b = False
        print('First line of text 3')
        print(target_text)

# Need to filter : keep only phrases where you know all words (keep the 30 000 most used words)


english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split(" ")])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split(" ")])
# print(french_sentences[0])
# print([word for word in french_sentences[0].split(" ")])

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = \
    preprocess(english_sentences, french_sentences, False)

max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index) + 1  # 0 padding

input_token_index = {word: id for word, id in english_tokenizer.word_index.items()}
target_token_index = {word: id for word, id in french_tokenizer.word_index.items()}
reverse_target_char_index = {id: word for word, id in french_tokenizer.word_index.items()}
print('Shape of preproc_english_sentences', preproc_english_sentences.shape)
print('Shape of preproc_french_sentences', preproc_french_sentences.shape)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)


tmp_x = pad(preproc_english_sentences, max_french_sequence_length)

X_train, X_test, y_train, y_test = train_test_split(english_sentences, french_sentences)
# X_train, X_test, y_train, y_test = train_test_split(tmp_x, preproc_french_sentences)

#1 X_train = X_train.reshape((-1, y_train.shape[-2], 1))
#1 X_test = X_test.reshape((-1, y_test.shape[-2], 1))
# X_train = X_train.reshape((-1, y_train.shape[-2]))
# X_test = X_test.reshape((-1, y_test.shape[-2]))


def generate_batch(X=X_train, y=y_train, batch_size=128):
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = np.zeros((batch_size, max_english_sequence_length), dtype='float32')
            decoder_input_data = np.zeros((batch_size, max_french_sequence_length), dtype='float32')
            decoder_target_data = np.zeros((batch_size, max_french_sequence_length, french_vocab_size), dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text.split(" ")):
                    try:
                        encoder_input_data[i, t] = input_token_index[word]
                    except:
                        pass

                for t, word in enumerate(target_text.split(" ")):
                    if t < len(input_text)-1:
                        try:
                            decoder_input_data[i, t] = target_token_index[word]
                        except:
                            pass
                    if t > 0:
                        try:
                            decoder_target_data[i, t - 1, target_token_index[word]] = 1.
                        except:
                            pass
            yield([encoder_input_data, decoder_input_data], decoder_target_data)


# model, encoder_model, decoder_model = encoder_decoder(english_vocab_size, french_vocab_size)
# HERE models
mode = 3
if mode == 0:
    model, encoder_model, decoder_model = encoder_decoderRMSProp(english_vocab_size, french_vocab_size)
else:
    if mode == 1:
        model, encoder_model, decoder_model = encoder_decoderAdam(english_vocab_size, french_vocab_size)
    else:
        if mode == 2:
            model, encoder_model, decoder_model = encoder_decoderAdamBiggerEmbed(english_vocab_size, french_vocab_size)
        else:
            if mode == 3:
                model, encoder_model, decoder_model = encoder_decoderAdamBiggerLSTMCapacity(english_vocab_size, french_vocab_size)
            else:
                if mode == 4:
                    model, encoder_model, decoder_model = encoder_decoderAdamFast(english_vocab_size, french_vocab_size)


print(model.summary())
train_samples = len(X_train)
val_samples = len(X_test)
batch_size = 128
epochs = 50

model.fit_generator(generator=generate_batch(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=train_samples/batch_size,
                    epochs=epochs,
                    validation_data=generate_batch(X_test, y_test, batch_size=batch_size),
                    validation_steps=val_samples/batch_size)

# model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

model.save("models/EncoderDecoderModel"+str(mode))
model.save_weights('models/EncoderDecoder_weights'+str(mode)+'.h5')
# model.load_weights('models/EncoderDecoder_weights.h5')


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['<start>']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '<end>' or
           len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


train_gen = generate_batch(X_train, y_train, batch_size=1)
k = -1
for i in range(10):
    k += 1
    (input_seq, _), _ = next(train_gen)
    decoded_sentence = decode_sequence(input_seq)
    print('\nInput English sentence:', X_train[i])
    print('Actual French Translation:', y_train[i])
    print('Predicted French Translation:', decoded_sentence)
