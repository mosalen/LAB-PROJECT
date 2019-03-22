#! -*- coding: utf-8 -*-

import tensorflow as tf
import tokenization
from bert_serving.client import BertClient
from modeling import attention_layer


review_dir= 'D:/TBdata/'
doc_number = 100
bc = BertClient()
reviews = []

i=0
if i <= doc_number:

    print('(1) loading review doc'+str(i))
    reviews[i] = ['hello world', 'how are you?', 'I am great']
    #reviews = open(review_dir+'train'+str(i)+'.txt', encoding='utf-8').read().split('&')
    vec=[]
    doc_vec = []

    print('(2) converting sentences')

    j=0
    if j <= len(reviews):
        vec[j] = bc.encode(reviews[j])
        '''对vec进行self_attention计算'''
        doc_vec = tf.concat(vec[j], axis=-1)
        j+=1
    else:
        print('(3) doc_vecs created')

        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)

    with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
            attention_head = attention_layer(
                from_tensor=doc_vec,
                to_tensor=doc_vec,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=0.0,
                initializer_range=0.02,
                do_return_2d_tensor=False,
                batch_size=batch_size,
                from_seq_length=seq_length,
                to_seq_length=seq_length)
            attention_heads.append(attention_head)

        attention_output = attention_heads[0]




    i += 1
else:
    print('all docs have been analyzed. mission completed.')