# Seq2Seq_pretrained

## Introduction
This code apply the pretrained BERT encoder to the Seq2Seq model.

Encoder: borrow some code from the project **pytorch_pretrained_bert**, these maybe the early version code and now the project has changed to the [transformers](https://github.com/huggingface/transformers), but the old version code in this project is still working.

Decoder: borrow some code from the [ONMT](https://github.com/OpenNMT/OpenNMT-py).

The core codes of this project are the class `Seq2Seq` and `BertForSequenceGeneration` in the `./pytorch_pretrained_bert/modeling.py`, which combine the BERT encoder and ONMT decoder.
