from loader import load_data
import collections
import gensim


english_sentences = load_data('data/small_vocab_en')
french_sentences = load_data('data/small_vocab_fr')
english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

# print(preproc_english_sentences)
# print(english_sentences)
word2VecModel = gensim.models.Word2Vec(sentences=[s.split(' ') for s in english_sentences], size=256, min_count=1)
print(list(word2VecModel.wv.vocab))
print(word2VecModel.wv.most_similar('june'))

