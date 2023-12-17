#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
from dataclasses import dataclass
import torch
import logging
import random
import math
from typing import Optional, Union
from collections import OrderedDict
from transformers import PreTrainedTokenizerBase, PreTrainedModel
# from transformers.file_utils import PaddingStrategy

from extraction.record_schema import RecordSchema
from extraction.dataset_processer import spot_prompt, asoc_prompt
from extraction.constants import BaseStructureMarker, text_start
from extraction.utils import convert_to_record_function
from extraction.noiser.spot_asoc_noiser import SpotAsocNoiser


logger = logging.getLogger("__main__")


class DynamicSSIGenerator():
    """
    Sample negative spot and asoc to construct SSI
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, schema: RecordSchema, positive_rate=1, negative=5, ordered_prompt=False) -> None:
        self.spot_dict = self.get_ordered_dict(schema.type_list, tokenizer)
        # print(self.spot_dict)
        self.asoc_dict = self.get_ordered_dict(schema.role_list, tokenizer)
        self.spot_asoc_dict = schema.type_role_dict
        self.spot_list = list(self.spot_dict.keys())
        self.asoc_list = list(self.asoc_dict.keys())
        self.spot_prompt = tokenizer.get_vocab()[spot_prompt]
        self.asoc_prompt = tokenizer.get_vocab()[asoc_prompt]
        self.text_start = tokenizer.get_vocab()[text_start]
        self.positive_rate = positive_rate if positive_rate > 0 and positive_rate < 1 else 1
        self.negative = negative
        self.ordered_prompt = ordered_prompt
        logger.info(f"Meta Sample, Negative: {self.negative}, Ordered Prompt: {self.ordered_prompt}")

    @staticmethod
    def get_ordered_dict(schema_name_list, tokenizer):
        schema_ordered_dict = OrderedDict()
        for name in schema_name_list:
            schema_ordered_dict[name] = tokenizer.encode(name, add_special_tokens=False)
        return schema_ordered_dict

    @staticmethod
    def sample_negative(postive, candidates, k=5):
        if k < 0:
            k = len(candidates)
        negative_set = set()
        for index in torch.randperm(len(candidates))[:k].tolist():
            negative = candidates[index]
            if negative not in postive:
                negative_set.add(negative)
        return list(negative_set)

    def sample_spot(self, positive):
        """ Sample spot
        """
        negative_spot = self.sample_negative(postive=positive, candidates=self.spot_list, k=self.negative)
        positive_spot = random.sample(positive, math.floor(len(positive) * self.positive_rate))
        prefix_spot_candidates = positive_spot + negative_spot
        # print(prefix_spot_candidates)

        converted_spot_prefix = self.convert_prefix(
            candidates=prefix_spot_candidates,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=self.ordered_prompt,
        )

        return converted_spot_prefix, positive_spot, negative_spot

    def sample_asoc(self, positive):
        """ Sample Asoc
        """
        negative_asoc = self.sample_negative(postive=positive, candidates=self.asoc_list, k=self.negative)
        prefix_asoc_candidates = positive + negative_asoc
        converted_asoc_prefix = self.convert_prefix(
            candidates=prefix_asoc_candidates,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=self.ordered_prompt,
        )
        return converted_asoc_prefix, negative_asoc
    
    def sample_spot_asoc(self, positive):
        """ 
        Sample Asoc
        """
        prefix = list()
        positive_spot_asoc = list()
        negative_spot_asoc = list()
        negative_set = set()
        for index in self.spot_list:
            if index not in positive:
                negative_set.add(index)
        prefix_spot_asoc_candidates = positive + list(negative_set)
        
        for spot in prefix_spot_asoc_candidates:
            prefix += [self.spot_prompt]
            prefix += self.spot_dict[spot]
            if spot in positive:
                positive_spot_asoc += [self.spot_prompt]
                positive_spot_asoc += self.spot_dict[spot] 
                
            for asoc in self.spot_asoc_dict[spot]:
                prefix += [self.asoc_prompt]
                prefix += self.asoc_dict[asoc]
                if spot in positive:
                    positive_spot_asoc += [self.asoc_prompt]
                    positive_spot_asoc += self.asoc_dict[asoc] 
            
            
            
        return prefix, positive_spot_asoc, negative_spot_asoc

    def full_spot(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        return self.convert_prefix(
            candidates=self.spot_list,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=ordered_prompt,
        )

    def full_asoc(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        return self.convert_prefix(
            candidates=self.asoc_list,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=ordered_prompt,
        )

    @staticmethod
    def convert_prefix(candidates, prompt, mapper, ordered_prompt=True):
        prefix = list()

        if ordered_prompt:
            candidate_sorted = sorted([(candidate, index) for index, candidate in enumerate(candidates)])
            index_list = [index for _, index in candidate_sorted]
        else:
            index_list = torch.randperm(len(candidates)).tolist()

        # print(candidates)
        for index in index_list:
            prefix += [prompt]
            prefix += mapper[candidates[index]]
        return prefix


@dataclass
class DataCollatorForMetaSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        max_target_length (:obj:`int`, `optional`):
            Maximum length of target sequence length.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """
    tokenizer: PreTrainedTokenizerBase
    negative_sampler: dict
    model: Optional[PreTrainedModel] = None
    padding: bool =  True
    max_length: Optional[int] = None
    max_target_length: Optional[int] = None
    max_prefix_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    spot_asoc_nosier: SpotAsocNoiser = None
    decoding_format: str = 'spotasoc'

    def __call__(self, features):
        """ Make Meta Schema Batch

        Args:
            features (Dict): [description]
                - sample_prompt: indicates sample_prompt example, need pop after call
                - spots (List[str]): List of spots in this sentence, need pop after call
                - asocs (List[str]): List of asocs in this sentence, need pop after call
                - input_ids
                - attention_mask
                - labels

        Returns:
        """
        batch_entry = {}
        B = len(features)
        if "fc_feats" in features[0]:
            V_L = max(entry['boxes'].shape[0] for entry in features)
            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            images = torch.zeros(B, 3, 224 ,224, dtype=torch.float)
            fc_dim = features[0]['fc_feats'].shape[-1]
            fc_feats = torch.zeros(B, V_L,fc_dim, dtype=torch.float)
            vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)
            attr_dim = features[0]['attr_feats'].shape[-1]
            attr_feats = torch.zeros(B, V_L,fc_dim, dtype=torch.float)

        
        targets = []
        img_ids = []
        img_paths = []
        input_text = []   
         
        for i,feature in enumerate(features):

            sample_prompt = feature['sample_prompt']

            if not sample_prompt:
                # Evaluation using Ordered SSI
                converted_spot_prefix = self.negative_sampler.full_spot(shuffle=self.model.training)
                converted_asoc_prefix = self.negative_sampler.full_asoc(shuffle=self.model.training)
            else:
                # Sample SSI
                # import pdb 
                # pdb.set_trace()
                # print(feature.get('asoc', []))
                task =  feature['task']
                converted_spot_prefix, positive_spot, negative_spot = self.negative_sampler[task].sample_spot(positive=feature.get('spots', []))
                converted_asoc_prefix, negative_asoc = self.negative_sampler[task].sample_asoc(positive=feature.get('asocs', []))
                spot_asoc_prefix, positive_spot_asoc, negative_spot_asoc = self.negative_sampler[task].sample_spot_asoc(positive=feature.get('spots', []))
                # converted_asoc_prefix, negative_asoc = self.negative_sampler.sample_asoc('asocs', [])
                    

                # Dynamic generating spot-asoc during training
                if 'spot_asoc' in feature:

                    # Deleted positive example Spot in Target that was not sampled by Prefix
                    # print(positive_spot)
                    feature['spot_asoc'] = [spot_asoc for spot_asoc in feature['spot_asoc'] if spot_asoc["label"] in positive_spot]
                    if self.spot_asoc_nosier is not None and feature["mode"] == "train":
                        if isinstance(self.spot_asoc_nosier, SpotAsocNoiser):
                            feature['spot_asoc'] = self.spot_asoc_nosier.add_noise(
                                feature['spot_asoc'],
                                spot_label_list=negative_spot,
                                asoc_label_list=negative_asoc,
                            )
                        else:
                            raise NotImplementedError(f'{self.spot_asoc_nosier} is not implemented.')

                    # Generate new record
                    record = convert_to_record_function[self.decoding_format](
                        feature['spot_asoc'],
                        structure_maker=BaseStructureMarker(),
                        task = feature['task']
                    )
                    feature["labels"] = self.tokenizer.encode(record)

            if feature["task"].endswith("arg"):
                prefix = positive_spot_asoc
            elif feature["task"].endswith("trigger"):
                prefix = converted_spot_prefix
            else:
                prefix = converted_spot_prefix + converted_asoc_prefix
                
            tmp_prefix = prefix
            # generated_sents = self.tokenizer.batch_decode(tmp_prefix, skip_special_tokens=False,clean_up_tokenization_spaces=False)
            # print(feature["task"], generated_sents)

            

            if "fc_feats" in feature:     
                n_boxes = feature['n_boxes']
            # print(n_boxes)
                boxes[i,:n_boxes] = feature['boxes'] if 'boxes' in feature else print(i)
                # images[i] = feature['image']
                fc_feats[i,:n_boxes] = feature['fc_feats']
                # print(feature['vis_feats'].shape,  vis_feats[i,:].shape)
                vis_attention_mask[i, :n_boxes] = 1
                img_ids.append(feature['img_id'])        

                   
            input_text.append(feature['input_text'])            
            targets.append(feature['targets'])
            task = feature['task']
            
            
            feature.pop('mode') if 'mode' in feature else None
            feature.pop('sample_prompt') if 'sample_prompt' in feature else None
            feature.pop('spot_asoc') if 'spot_asoc' in feature else None
            feature.pop('spots') if 'spots' in feature else None
            feature.pop('asocs') if 'asocs' in feature else None
            feature.pop('img_id') if 'img_id' in feature else None
            feature.pop('boxes') if 'boxes' in feature else None            
            feature.pop('input_text') if 'input_text' in feature else None
            feature.pop('input_length') if 'input_length' in feature else None
            feature.pop('sent') if 'sent' in feature else None
            feature.pop('labels_length') if 'labels_length' in feature else None

            feature.pop('args') if 'args' in feature else None
            feature.pop('fc_feats') if 'fc_feats' in feature else None
            feature.pop('n_boxes') if 'n_boxes' in feature else None
            feature.pop('targets') if 'targets' in feature else None
            feature.pop('attr_feats')  if 'attr_feats' in feature else None
            feature.pop('image')  if 'image' in feature else None
            feature.pop('task')  if 'task' in feature else None
            # prefix = []
            if self.max_prefix_length is not None and self.max_prefix_length >= 0:
                prefix = prefix[:self.max_prefix_length]

            feature['input_ids'] =  feature['input_ids'] + prefix 
            # feature['input_ids'] = prefix + [self.negative_sampler.text_start] + feature['input_ids']

            

            
       

            # truncate `input_ids` to max length
            if self.max_length:
                feature['input_ids'] = feature['input_ids'][:self.max_length]
            if self.max_target_length and 'labels' in feature:
                feature['labels'] = feature['labels'][:self.max_target_length]

            feature['attention_mask'] = [1] * len(feature['input_ids'])
            
          



              
        batch_entry['input_text'] = input_text
        batch_entry['targets'] = targets
        batch_entry['boxes'] = boxes
        batch_entry['vis_feats'] = fc_feats

        batch_entry['vis_attention_mask'] = vis_attention_mask
        batch_entry['img_id'] = img_ids
        
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(_label) for _label in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
     
        features = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        # print(type(features.ke))
        # print(batch_entry.keys())
        batch_entry["input_ids"] = features["input_ids"]
        batch_entry["labels"] = features["labels"]
        batch_entry["attention_mask"] = features["attention_mask"]
        batch_entry["decoder_input_ids"] = features["decoder_input_ids"]
        batch_entry["task"] = task

        # batch_entry.update(feature)
        # print(batch_entry.keys())
        return batch_entry
