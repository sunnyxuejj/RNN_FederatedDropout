# Federated Dropout on RNN model

This repository extend [Federated Dropout](https://arxiv.org/abs/1812.07210) to recurrent connections of RNN models.

The RNN model comes with instructions to train word level language models over the [Reddit](https://github.com/TalwalkarLab/leaf/tree/master/data/reddit)

For RNN Dropout:
+  [A theoretically grounded application of dropout in recurrent neural networks](https://arxiv.org/abs/1512.05287) propose that every weight matrix of recurrent layers could be dropped out
+  We directly zero out partial weight rows from weight_ih and weight_hh matrices during local training of the client

The model can be composed of two LSTM layers.

+ Install PyTorch 1.12
+ Get Reddit dataset from https://github.com/TalwalkarLab/leaf/tree/master/data/reddit
+ Run `data/user_data.py` to preprocess data
+ Run `run_att.py`
