# LAB-PROJECT
A testing project for review spam detection.
This is a novel project based on Google BERT: https://github.com/google-research/bert.
However, we do not fine-tune the model for classification. Instead, we use the feature-based method. Also, we have also changed the pre-trained task that we will get public the source code of that part later.

# Sentence Encoder
We use bert-as-service to acquire sentence representation fast. Certainly, you may use the source code of BERT to get the representation as well. Please click https://bert-as-service.readthedocs.io/en/latest/section/what-is-it.html for more info.

# Run
BERT is a very huge unsupervised pre-trained model. So we recommend to run it on GPU or TPU, or it will be very slow. But the result will be surely outstanding.

# License
This is a working paper so please do not use it for any academic purpose at this stage. We sincerely thank you whereas all kinds of discussion are welcome. 

# Modules
Two modules, positioanl encoding & multi-head self-attention, are also implemented and utilized in the down-stream task (i.e., doc encoder), just like how we treat words in one sentence.
