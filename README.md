# mxnet-seq2seq

This project implements the sequence to sequence learning with mxnet for open-domain chatbot

## Sequence to Sequence learning with LSTM encoder-decoder

The seq2seq encoder-decoder architecture is introduced by [Sequence to Sequence Learning with Neural Networks](http://arxiv.org/abs/1409.3215)
 
This implementation borrows idea from **lstm_bucketing**, I slightly modified it and reconstructed the embedding layer.

## How to run

Firstly, process the data by
```
python datautils.py
```
then run the model by
```
python main.py
```

## The architecture

We know that **seq2seq encoder-decoder** architecture includes two RNNs (LSTMs), one for encoding source sequence and another for decoding target sequence.

For NLP-related tasks, the sequence could be a natural language sentence. As a result, the encoder and decoder should **share the word embedding layer** .

The bucketing is a grate solution adapting the arbitrariness of sequence length. I padding zero to a fixed length at the encoding sequence and make buckets at the decoding phrase. 

The data is formatted as:

```
0 0 ... 0 23 12 121 832 || 2 3432 898 7 323
0 0 ... 0 43 98 233 323 || 7 4423 833 1 232
0 0 ... 0 32 44 133 555 || 2 4534 545 6 767
---
0 0 ... 0 23 12 121 832 || 2 3432 898 7
0 0 ... 0 23 12 121 832 || 2 3432 898 7
0 0 ... 0 23 12 121 832 || 2 3432 898 7
---

```
The input shape for embedding layer is **(batch\_size, seq\_len)**, the input shape for lstm encoder is **(batch\_size, seq\_len, embed\_dim)** .



## More details coming soon 

For any question, please send me email. 

    yoosan.zhou at gmail dot com