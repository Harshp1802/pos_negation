import pickle
from utils import create_vocab, convert2index
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow import keras

np.random.seed(42)

training_sentences, training_POS, training_scope = pickle.load( open( "training_starsem_pos_scope.p", "rb" ) )
testing_sentences, testing_POS, testing_scope = pickle.load( open( "testing_starsem_pos_scope_1.p", "rb" ) )

words, tags, word2index, tag2index = create_vocab(training_sentences, training_POS)
index2tag = {value : key for (key, value) in tag2index.items()}
index2word = {value : key for (key, value) in word2index.items()}

train_sentences_X, train_tags_y = convert2index(training_sentences,word2index,training_POS,tag2index)
test_sentences_X, test_tags_y = convert2index(testing_sentences,word2index,testing_POS,tag2index)

MAX_LENGTH = len(max(train_sentences_X, key=len))

train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')
train_scope_y = pad_sequences(training_scope, maxlen=MAX_LENGTH, padding='post')
test_scope_y = pad_sequences(testing_scope, maxlen=MAX_LENGTH, padding='post')

out_classes1 = np.max(np.concatenate((train_tags_y,test_tags_y),axis=None))+1
out_classes2 = np.max(np.concatenate((train_scope_y,test_scope_y),axis=None))+1

y1_train = keras.utils.to_categorical(train_tags_y, out_classes1)
y1_test = keras.utils.to_categorical(test_tags_y, out_classes1)
y2_train = keras.utils.to_categorical(train_scope_y, out_classes2)
y2_test = keras.utils.to_categorical(test_scope_y, out_classes2)

num_words=len(word2index)
embedding_vector_length = 128
max_length=MAX_LENGTH
batch_size=32
epochs=50

def model1()