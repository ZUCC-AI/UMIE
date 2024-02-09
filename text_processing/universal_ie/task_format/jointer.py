#!/usr/bin/env python
# -*- coding:utf-8 -*-


import json
import ast
from typing import List
from universal_ie.utils import tokens_to_str, change_ptb_token_back
from universal_ie.ie_format import Entity, Label, Relation, Sentence, Span
from universal_ie.task_format.task_format import TaskFormat


class JointER(TaskFormat):
    """ Joint Entity Relation Data format at https://github.com/yubowen-ph/JointER"""

    def __init__(self, sentence_json, language='en'):
        super().__init__(
            language=language
        )
        self.tokens = sentence_json['tokens']
        for index in range(len(self.tokens)):
            self.tokens[index] = change_ptb_token_back(self.tokens[index])
        if self.tokens is None:
            print('[sentence without tokens]:', sentence_json)
            exit(1)
        self.h = sentence_json['h']
        self.t = sentence_json['t']
        self.relation = sentence_json['relation']


    @staticmethod
    def load_from_file(filename, language='en') -> List[Sentence]:
        sentence_list = list()
        raw_instance_list = json.load(open(filename))
        print(f"{filename}: {len(raw_instance_list)}")
        for instance in raw_instance_list:
            instance = JointER(
                    sentence_json=instance,
                    language=language
                ).generate_instance()
            sentence_list += [instance]
        return sentence_list

class MNRE(TaskFormat):
    """ Joint Entity Relation Data format at https://github.com/yubowen-ph/JointER"""

    def __init__(self, sentence_json, language='en'):
        super().__init__(
            language=language
        )
        self.tokens = sentence_json['token']
        for index in range(len(self.tokens)):
            self.tokens[index] = change_ptb_token_back(self.tokens[index])
        if self.tokens is None:
            print('[sentence without tokens]:', sentence_json)
            exit(1)
        self.h = sentence_json['h']
        self.t = sentence_json['t']
        self.relation = sentence_json['relation']
        self.image_id = sentence_json['img_id']

    def generate_instance(self):
        entities = dict()
        relations = dict()
        entity_map = dict()
        s_s,s_e = self.h['pos'] 
        s_t = "none"  
        tokens = self.tokens[s_s: s_e]
        indexes = list(range(s_s, s_e))
        if (s_s, s_e, s_t) not in entity_map:
            entities[(s_s, s_e, s_t)] = Entity(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                ),
                label=Label(s_t)
            )

        o_s, o_e = self.t["pos"]
        o_t = "none"
        tokens = self.tokens[o_s: o_e]
        indexes = list(range(o_s, o_e))
        if (o_s, o_e, o_t) not in entity_map:
            entities[(o_s, o_e, o_t)] = Entity(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                ),
                label=Label(o_t)
            )

        relations[0] = Relation(
            arg1=entities[(s_s, s_e, s_t)],
            arg2=entities[(o_s, o_e, o_t)],
            label=Label(self.relation),
        )

        return (Sentence(
            tokens=self.tokens,
            entities=entities.values(),
            relations=relations.values(),
        ),self.image_id)


    @staticmethod
    def load_from_file(filename, language='en') -> List[Sentence]:
        sentence_list = list()

        # raw_instance_list = json.load(open(filename))
        with open(filename, "r", encoding="utf-8") as f:
            raw_instance_list = f.readlines()
        
            print(f"{filename}: {len(raw_instance_list)}")

            for instance in raw_instance_list:
                instance = MNRE(
                        sentence_json=ast.literal_eval(instance),
                        language=language
                    ).generate_instance()
                # import pdb
                # pdb.set_trace()
                sentence_list += [instance]
            return sentence_list
