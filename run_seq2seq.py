from __future__ import absolute_import, division, print_function

import argparse
import codecs
import csv
import logging
import math
import os
import random
import time
from itertools import count

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import onmt.decoders.ensemble
import onmt.inputters as inputters
import onmt.model_builder
import onmt.translate.beam
from onmt.modules.copy_generator import collapse_copy_scores
from onmt.translate.beam_search import BeamSearch
from onmt.translate.random_sampling import RandomSampling
from onmt.utils.misc import set_random_seed, tile
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (BertForSequenceClassification,
                                              BertForSequenceGeneration,
                                              Seq2Seq)
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def batch_sort(batch):
    '''
    batch: src_ids, src_mask, segment_ids, tgt_ids, tgt_mask
    '''
    # time0 = time.time()
    src_length = batch[1].sum(1)
    _, index = src_length.sort(descending=True)
    batch = [var[index] for var in batch]
    time1 = time.time()
    # print(time1 - time0)
    return batch


class InputExample(object):
    def __init__(self, guid, text_a, text_b):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    def __init__(self, src_ids, src_mask, segment_ids, tgt_ids, tgt_mask):
        self.src_ids = src_ids
        self.src_mask = src_mask
        self.segment_ids = segment_ids
        self.tgt_ids = tgt_ids
        self.tgt_mask = tgt_mask


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
        Reads a tab separated value file.
        src_text \t tgt_text
        """
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod       
    def _read_summ_data(cls, input_file):
        lines = list(open(input_file, 'r', encoding='utf8').readlines())
        lines = [line.strip().split('\t') for line in lines]
        return lines


class GigaProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        return self._create_examples(self._read_summ_data(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
        return self._create_examples(self._read_summ_data(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        return self._create_examples(self._read_summ_data(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) == 2:
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                text_b = line[1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples


class CNNDMProcessor(DataProcessor):
    pass


def covert_examples_to_features(examples, max_src_length, max_tgt_length, src_tokenizer, tgt_tokenizer):
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc='loading')):
        tokens_a = src_tokenizer.tokenize(example.text_a)
        tokens_b = tgt_tokenizer.tokenize(example.text_b)
        if len(tokens_a) > max_src_length - 2:
            tokens_a = tokens_a[:(max_src_length - 2)]
        if len(tokens_b) > max_tgt_length - 2:
            tokens_b = tokens_b[:(max_tgt_length - 2)]
        
        tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
        tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]

        # only token_src needed
        segment_ids = [0] * len(tokens_a)

        src_ids = src_tokenizer.convert_tokens_to_ids(tokens_a)
        tgt_ids = tgt_tokenizer.convert_tokens_to_ids(tokens_b)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        src_mask = [1] * len(src_ids)
        tgt_mask = [1] * len(tgt_ids)

        src_padding = [0] * (max_src_length - len(src_mask))
        tgt_padding = [0] * (max_tgt_length - len(tgt_mask))

        src_ids += src_padding
        src_mask += src_padding   
        segment_ids += src_padding

        tgt_ids += tgt_padding
        tgt_mask += tgt_padding

        assert len(src_ids) == max_src_length
        assert len(src_mask) == max_src_length
        assert len(segment_ids) == max_src_length
        assert len(tgt_ids) == max_tgt_length
        assert len(tgt_mask) == max_tgt_length

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("src tokens a: %s" % " ".join(
                    [str(x) for x in tokens_a]))
            logger.info("src tokens b: %s" % " ".join(
                    [str(x) for x in tokens_b]))
            logger.info("src_ids: %s" % " ".join([str(x) for x in src_ids]))
            logger.info("tgt_ids: %s" % " ".join([str(x) for x in tgt_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in src_mask]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in tgt_mask]))

        features.append(
            InputFeatures(
                src_ids=src_ids,
                src_mask=src_mask,
                segment_ids=segment_ids,
                tgt_ids=tgt_ids,
                tgt_mask=tgt_mask
            )
        )

    return features


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def rouge():
    return 0


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--model_type",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the model type, gru or transformer.")

    
    ## Other parameters
    parser.add_argument("--max_src_length",
                        default=400,
                        type=int,
                        help="The maximum total src sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--max_tgt_length",
                        default=100,
                        type=int,
                        help="The maximum total tgt sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run train.")

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval.")

    parser.add_argument("--do_infer",
                        action='store_true',
                        help="Whether to run eval.")

    parser.add_argument("--checkpoint",
                        action='store_true',
                        help="Whether to save checkpoint every epoch.")

    parser.add_argument("--checkpoint_id", default=-1, type=int, help="the checkpoint to eval or infer")
                    
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    
    parser.add_argument("--eval_batch_size",
                        default=20,
                        type=int,
                        help="Total batch size for evaling.")

    parser.add_argument("--infer_batch_size",
                        default=20,
                        type=int,
                        help="Total batch size for infering.")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument("--infer_max_steps",
                        default=20,
                        type=int,
                        help="max step for inference.")

    parser.add_argument("--infer_min_steps",
                        default=0,
                        type=int,
                        help="min step for inference.")


    args = parser.parse_args()

    # data processor
    processors = {
        "giga": GigaProcessor,
        "cnndm": CNNDMProcessor,
    }


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_infer:
        raise ValueError("At least one of `do_train` or `do_eval` or 'do_infer' must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))


    # 实例化processor类
    processor = processors[task_name]()

    # 实例化tokenizer
    src_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tgt_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = Seq2Seq.from_pretrained(args.bert_model, model_type=args.model_type)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)


    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)


    ## do train
    global_step = 0
    nb_tf_steps = 0
    tf_loss = 0
    if args.do_train:
        train_features = covert_examples_to_features(
            examples=train_examples,
            max_src_length=args.max_src_length,
            max_tgt_length=args.max_tgt_length,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_src_ids = torch.tensor([f.src_ids for f in train_features], dtype=torch.long)
        all_src_mask = torch.tensor([f.src_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_tgt_ids = torch.tensor([f.tgt_ids for f in train_features], dtype=torch.long)
        all_tgt_mask = torch.tensor([f.tgt_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_src_ids, all_src_mask, all_segment_ids, all_tgt_ids, all_tgt_mask)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                batch = batch_sort(batch)
                src_ids, src_mask, segment_ids, tgt_ids, tgt_mask = batch
                loss, _, _ = model(src_ids, src_mask, segment_ids, tgt_ids, tgt_mask)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += src_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if args.checkpoint:
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(args.output_dir, args.task_name + "_" + args.model_type + "_" + str(epoch) + "_pytorch_model.bin")
                if args.do_train:
                    torch.save(model_to_save.state_dict(), output_model_file)

    # Save a trained model
    if args.do_train:
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, args.task_name + "_" + args.model_type + "_pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

    if args.checkpoint_id == -1:
        output_model_file = os.path.join(args.output_dir, args.task_name + "_" + args.model_type + "_pytorch_model.bin")
    else:
        output_model_file = os.path.join(args.output_dir, args.task_name + "_" + args.model_type + "_" + str(args.checkpoint_id) + "_pytorch_model.bin")
            
    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file)
    model = Seq2Seq.from_pretrained(args.bert_model, state_dict=model_state_dict, model_type=args.model_type)
    model.to(device)

    ## do eval
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0): 
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = covert_examples_to_features(
            examples=eval_examples,
            max_src_length=args.max_src_length,
            max_tgt_length=args.max_tgt_length,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer
        )
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_src_ids = torch.tensor([f.src_ids for f in eval_features], dtype=torch.long)
        all_src_mask = torch.tensor([f.src_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_tgt_ids = torch.tensor([f.tgt_ids for f in eval_features], dtype=torch.long)
        all_tgt_mask = torch.tensor([f.tgt_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_src_ids, all_src_mask, all_segment_ids, all_tgt_ids, all_tgt_mask)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_rouge = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            batch = batch_sort(batch)
            src_ids, src_mask, segment_ids, tgt_ids, tgt_mask = batch

            with torch.no_grad():
                tmp_eval_loss, _, _ = model(src_ids, src_mask, segment_ids, tgt_ids, tgt_mask)
                # print(tmp_eval_loss)

            tgt_ids = tgt_ids.to('cpu').numpy()
            tmp_eval_rouge = rouge()

            eval_loss += tmp_eval_loss.mean().item()
            eval_rouge += tmp_eval_rouge

            nb_eval_examples += src_ids.size(0)
            nb_eval_steps += 1
        
        eval_loss = eval_loss / nb_eval_steps
        eval_rouge = eval_rouge / nb_eval_examples
        result = {
            'eval_loss': eval_loss,
            'eval_rouge': eval_rouge,
            'global_step': global_step,
        }

        output_eval_file = os.path.join(args.output_dir, str(args.checkpoint_id) + "_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info(" %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    ## do infer
    if args.do_infer and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        final_output = []
        infer_examples = processor.get_test_examples(args.data_dir)
        infer_features = covert_examples_to_features(
            examples=infer_examples,
            max_src_length=args.max_src_length,
            max_tgt_length=args.max_tgt_length,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer
        )
        logger.info("***** Running inference *****")
        logger.info("  Num examples = %d", len(infer_examples))
        logger.info("  Batch size = %d", args.infer_batch_size)
        all_src_ids = torch.tensor([f.src_ids for f in infer_features], dtype=torch.long)
        all_src_mask = torch.tensor([f.src_mask for f in infer_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in infer_features], dtype=torch.long)
        all_tgt_ids = torch.tensor([f.tgt_ids for f in infer_features], dtype=torch.long)
        all_tgt_mask = torch.tensor([f.tgt_mask for f in infer_features], dtype=torch.long)
        infer_data = TensorDataset(all_src_ids, all_src_mask, all_segment_ids, all_tgt_ids, all_tgt_mask)

        infer_sampler = SequentialSampler(infer_data)
        infer_dataloader = DataLoader(infer_data, sampler=infer_sampler, batch_size=args.infer_batch_size)

        model.eval()
        eval_loss, eval_rouge = 0, 0
        eval_loss, eval_rouge = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in tqdm(infer_dataloader, desc="infering"):
            batch = tuple(t.to(device) for t in batch)
            batch = batch_sort(batch)
            src_ids, src_mask, segment_ids, tgt_ids, tgt_mask = batch

            with torch.no_grad():
                src_ids = src_ids.transpose(0, 1).unsqueeze(2)
                tgt_ids = tgt_ids.transpose(0, 1).unsqueeze(2)

                lengths = src_mask.sum(1)

                max_src_length = src_ids.size(0)
                max_tgt_length = tgt_ids.size(0)

                enc_state, memory_bank, lengths = model.encoder(src_ids, lengths)

                all_decoder_outputs = torch.zeros(args.infer_batch_size, args.infer_max_steps)
                all_attention_outputs = torch.zeros(args.infer_max_steps, args.infer_batch_size, max_src_length)

                all_decoder_outputs = all_decoder_outputs.to(device)
                all_attention_outputs = all_attention_outputs.to(device)

                model.decoder.init_state(src_ids, memory_bank, enc_state)

                decoder_input = torch.LongTensor([101] * args.infer_batch_size)
                decoder_input = decoder_input.to(device)
                decoder_input = decoder_input.unsqueeze(0)
                decoder_input = decoder_input.unsqueeze(2)

                for step in range(args.infer_max_steps):
                    dec_out, dec_attn = model.decoder(decoder_input, memory_bank, memory_lengths=lengths, step=step)
                    logits = model.generator(dec_out)

                    if step + 1 < args.infer_min_steps:
                        for i in range(logits.size(1)):
                            logits[0][i][102] = -1e20

                    prob, idx = torch.max(logits, 2)
                    decoder_input = idx.unsqueeze(2)

                    all_decoder_outputs[:, step] = idx.squeeze(0)
                    # all_attention_outputs[step, :, :] = dec_attn.squeeze(0)

                src_ids = src_ids.squeeze(2).transpose(0, 1)
                tgt_ids = tgt_ids.squeeze(2).transpose(0, 1)
                src_ids = src_ids.cpu().int().detach().numpy()
                tgt_ids = tgt_ids.cpu().int().detach().numpy()
                all_decoder_outputs = all_decoder_outputs.cpu().int().detach().numpy()

                for i in range(args.infer_batch_size):
                    src_text = src_tokenizer.convert_ids_to_tokens(src_ids[i])
                    tgt_text = tgt_tokenizer.convert_ids_to_tokens(all_decoder_outputs[i])
                    ref_text = tgt_tokenizer.convert_ids_to_tokens(tgt_ids[i])
                    final_output.append((tgt_text, ref_text))
        
        # out put infer file
        output_infer_file = os.path.join(args.output_dir, str(args.checkpoint_id) + "_infer_results.txt")
        with open(output_infer_file, 'w', encoding='utf8') as wtf:
            for line1, line2 in final_output:
                wtf.write(' '.join(line1) + '\t' + ' '.join(line2) + '\n')
        print('infering end')


if __name__ == "__main__":
    main()
