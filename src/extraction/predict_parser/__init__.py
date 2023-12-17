#!/usr/bin/env python
# -*- coding:utf-8 -*-
from extraction.predict_parser.predict_parser import PredictParser
from extraction.predict_parser.spotasoc_predict_parser import EntityPredictParser, RelationPredictParser,SWiGPredictParser, EventPredictParser


decoding_format_dict = {
    'entity': EntityPredictParser,
    'relation': RelationPredictParser,
    'swig': SWiGPredictParser,
    'event': EventPredictParser,
    'event_trigger': EntityPredictParser,
    'event_arg': SWiGPredictParser,
}


def get_predict_parser(decoding_schema, label_constraint):
    return decoding_format_dict[decoding_schema](label_constraint=label_constraint)
