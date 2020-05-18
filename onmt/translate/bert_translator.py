#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import time
from itertools import count

import torch

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.decoders.ensemble
from onmt.translate.beam_search import BeamSearch
from onmt.translate.random_sampling import RandomSampling
from onmt.utils.misc import tile, set_random_seed
from onmt.modules.copy_generator import collapse_copy_scores


class Translator(object):
    def __init__(
            self,
            model,
            src_reader,
            tgt_reader,
            gpu=-1,
            n_best=1,
            min_length=0,
            max_length=100,
            beam_size=30,
            random_sampling_topk=1,
            random_sampling_temp=1,
            stepwise_penalty=None,
            dump_beam=False,
            block_ngram_repeat=0,
            ignore_when_blocking=frozenset(),
            replace_unk=False,
            data_type="text",
            verbose=False,
            report_bleu=False,
            report_rouge=False,
            report_time=False,
            copy_attn=False,
            global_scorer=None,
            out_file=None,
            report_score=True,
            logger=None,
            seed=-1):
        self.model = model
        # self.fields = fields
        # tgt_field = dict(self.fields)["tgt"].base_field
        # self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = 102 # [SEP] 102
        self._tgt_pad_idx = 0 # [PAD] 0
        self._tgt_bos_idx = 101 # [CLS] 101
        self._tgt_unk_idx = 100 # [UNK] 100
        self._tgt_vocab_len = 30522

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = torch.device("cuda", self._gpu) \
            if self._use_cuda else torch.device("cpu")

        self.n_best = n_best
        self.max_length = max_length

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.replace_unk = replace_unk

        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError(
                "replace_unk requires an attentional decoder.")
        self.data_type = data_type
        self.verbose = verbose
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.report_time = report_time

        self.copy_attn = copy_attn

        self.global_scorer = global_scorer
        if self.global_scorer.has_cov_pen and \
                not self.model.decoder.attentional:
            raise ValueError(
                "Coverage penalty requires an attentional decoder.")
        self.out_file = out_file
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

        set_random_seed(seed, self._use_cuda)

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

            