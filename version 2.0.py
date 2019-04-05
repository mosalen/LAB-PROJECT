import tensorflow as tf
import numpy as np
from bert_serving.client import BertClient
from tensorflow.keras import backend as Backend
from scipy.special import softmax
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from self_attention import MultiHeadSelfAttention
from position import AddPositionalEncoding, TransformerCoordinateEmbedding


bc = BertClient()
MAX_SENT_LENGTH = 100
MAX_SENTS = 18
VALIDATION_SPLIT = 0.2

def Upstream(text):
    """generating sentence representation.
    supported by bert-as-service.
    source: https://bert-as-service.readthedocs.io"""
    text = text.split('#')
    vec = bc.encode(text)
    pad = np.zeros((MAX_SENTS, 768))
    pad[:vec.shape[0], :vec.shape[1]] = vec
    pad = pad.reshape((1, MAX_SENTS, 768))
    sen_vecs = np.array(pad).copy()
    print("generating sen representation, shape=" + str(np.shape(sen_vecs)))
    return sen_vecs


print('(1) pre-processing train texts...')
train_texts = open('//train_file_path/train.txt', encoding='utf-8').read().split('\n')

train_docs = np.empty((0, MAX_SENTS, 768))
for t_line in train_texts:
    t_line = Upstream(t_line)
    train_docs = np.append(arr=train_docs, values=t_line, axis=0)
print(' train_docs finished! shape is:'+str(np.shape(train_docs)))
tf_train=tf.convert_to_tensor(train_docs, dtype=tf.float32)

print('(2) pre-processing val texts...')
val_texts = open('//val_file_path/val.txt', encoding='utf-8').read().split('\n')

val_docs = np.empty((0, MAX_SENTS, 768))
for v_line in val_texts:
    v_line = Upstream(v_line)
    val_docs = np.append(arr=val_docs, values=v_line, axis=0)
print(' val_docs finished! shape is:'+str(np.shape(val_docs)))
tf_val=tf.convert_to_tensor(val_docs, dtype=tf.float32)


train_labels = open('//train_label_path/train_label.txt', encoding='utf-8').read().split('\n')
val_labels = open('//test_label_path/test_label.txt', encoding='utf-8').read().split('\n')
#all_texts = train_texts + test_texts
#all_labels = train_labels + test_labels
train_labels = to_categorical(np.asarray(train_labels))
val_labels = to_categorical(np.asarray(val_labels))

print('(3) building downstream model...')
doc_input = Input(shape=(MAX_SENTS, 768), dtype='float32')
pos = AddPositionalEncoding()(doc_input)
doc_encoder = MultiHeadSelfAttention(num_heads=1, use_masking=False)(pos)
flat = Flatten()(doc_encoder)
dense = Dense(384, activation='relu')(flat)
drop = Dropout(0.1)(dense)
pred = Dense(2, activation='sigmoid')(drop)
model = Model(doc_input, pred)
model.summary()
plot_model(model, to_file='//model_img_save_path/model.png',show_shapes=True)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
print (model.metrics_names)
model.fit(tf_train, train_labels, validation_data=(tf_val, val_labels), epochs=2, steps_per_epoch=64, validation_steps=64)
