# Seq2Seq_pretrained

## Introduction
This code apply the pretrained BERT encoder to the Seq2Seq model.

Encoder: borrow some code from the project **pytorch_pretrained_bert**, these maybe the early version code and now the project has changed to the [transformers](https://github.com/huggingface/transformers), but the old version code in this project is still working.

Decoder: borrow some code from the [ONMT](https://github.com/OpenNMT/OpenNMT-py).

The core codes of this project are the class `Seq2Seq` and `BertForSequenceGeneration` in the `./pytorch_pretrained_bert/modeling.py`, which combine the BERT encoder and ONMT decoder.

This project is the basic framework for the [paper](https://doi.org/10.1007/978-3-030-32381-3_14). 

## Get Started
1. Download the pretrained BERT model 'uncased_L-12_H-768_A-12'.
2. Download the [CNN & DM](https://github.com/harvardnlp/sent-summary) dataset and unzip to `./data`.
3. Preprocess data: `python cnndm_preprocessed.py`.
4. Train and infer:
```
Run seq2seq
## train
CUDA_VISIBLE_DEVICES=1 python run_seq2seq.py --data_dir=/home/rwei/All_data/cnndm --bert_model=uncased_L-12_H-768_A-12 --task_name=giga --output_dir=output/cnndm/seq2seq_transformer --model_type=transformer --do_lower_case --train_batch_size=10 --eval_batch_size=10 --infer_batch_size=10 --num_train_epochs=6 --max_src_length=400 --checkpoint --do_train
## infer
CUDA_VISIBLE_DEVICES=0 python run_seq2seq.py --data_dir=/home/rwei/All_data/cnndm --bert_model=uncased_L-12_H-768_A-12 --task_name=giga --output_dir=output/giga/seq2seq_transformer --model_type=transformer --do_lower_case --train_batch_size=80 --eval_batch_size=20 --infer_batch_size=20 --num_train_epochs=6 --checkpoint_id=0 --do_infer

Run BERT Seq2Seq
## train
CUDA_VISIBLE_DEVICES=0 python run_bert.py --data_dir=/home/rwei/All_data/cnndm --bert_model=uncased_L-12_H-768_A-12 --task_name=cnndm --output_dir=output/cnndm/bert_transformer --model_type=transformer --do_lower_case --train_batch_size=8 --eval_batch_size=20 --infer_batch_size=20 --num_train_epochs=6 --checkpoint --do_train
## infer
CUDA_VISIBLE_DEVICES=0 python run_bert.py --data_dir=/home/rwei/All_data/cnndm --bert_model=uncased_L-12_H-768_A-12 --task_name=giga --output_dir=output/giga/bert_transformer --model_type=transformer --do_lower_case --train_batch_size=50 --eval_batch_size=20 --infer_batch_size=20 --num_train_epochs=6 --checkpoint_id=0 --do_infer
```
5. Process the results to calculate Rouge:
```
python GigaRougeBuilder.py --eval_path=/root/eval --ref_path=output/cnndm/seq2seq_ref.txt --output_path=output/cnndm/seq2seq_gru/processed_5_infer_results.txt
```
6. Calculate Rouge scores through Perl scripts in server 105 & 209:
```
cd ROUGE
cd RELEASE-1.5.5
./ROUGE-1.5.5.pl -e data -a -m -n 2 /root/eval/ROUGE.xml >& /root/eval/giga_seq2seq_gru5_result.txt
cd ..
cd ..
```
