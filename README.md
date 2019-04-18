# LAB-PROJECT
A testing project for review spam detection.
This is a novel project based on Google BERT: https://github.com/google-research/bert.
However, we do not fine-tune the model for classification. Instead, we use the feature-based method. Also, we have also changed the pre-trained task that we will get public the source code of that part later.

# Sentence Encoder
We use bert-as-service to acquire sentence representation fast. Certainly, you may use the source code of BERT to get the representation as well. Please click https://bert-as-service.readthedocs.io/en/latest/section/what-is-it.html for more info.

Since the encoding stage could be quite slow (even with TPU), one may want to save the encoded files for the convenience of future training, so that this slow stage could be processed just once and skipped later. Therefore, we recommend to save the train_docs.npy and val_docs.npy after encoding and use np.load for later downstream tasks.

# Run
BERT is a very huge unsupervised pre-trained model. So we recommend to run it on GPU or TPU, or it will be very slow. But the result will be surely outstanding.

# On TPU
We have implemented the model on Colab and uploaded our Chinese-version BERT to our TPU, while we will get it public later. 
Note that currently there are four pre-trained BERT model checkpoints on google TPU:
uncased_L-12_H-768_A-12: uncased BERT base model; 
uncased_L-24_H-1024_A-16: uncased BERT large model; 
cased_L-12_H-768_A-12: cased BERT base model;
cased_L-24_H-1024_A-16: cased BERT large model;

load one of them in your training on TPU via:
BERT_MODEL = 'uncased_L-12_H-768_A-12' #@param {type:"string"}
BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL

# License
This is a working paper so please do not use it for any academic purpose at this stage. We sincerely thank you whereas all kinds of discussion are welcome. 

# Modules
Two modules, positioanl encoding & multi-head self-attention, are also implemented and utilized in the down-stream task (i.e., doc encoder), just like how we treat words in one sentence.
