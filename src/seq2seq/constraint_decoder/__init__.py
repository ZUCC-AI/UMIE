#!/usr/bin/env python
# -*- coding:utf-8 -*-
from seq2seq.constraint_decoder.spotasoc_constraint_decoder import (
    SpotAsocConstraintDecoder,
    SpotConstraintDecoder
)


def get_constraint_decoder(tokenizer, type_schema, decoding_schema, task_name='event', source_prefix=None):
    if decoding_schema == 'spotasoc':
        if len(type_schema.role_list) == 0:
            task_map = {
                'entity': SpotConstraintDecoder,
                'relation': SpotConstraintDecoder,
                'event': SpotConstraintDecoder,
                'record': SpotConstraintDecoder,
                'event_trigger': SpotConstraintDecoder,
                'event_trigger': SpotConstraintDecoder
            }
        else:
            task_map = {
                'entity': SpotAsocConstraintDecoder,
                'relation': SpotAsocConstraintDecoder,
                'event': SpotAsocConstraintDecoder,
                'record': SpotAsocConstraintDecoder,
                'event_trigger': SpotAsocConstraintDecoder
            }
    else:
        raise NotImplementedError(
            f'Type Schema {type_schema}, Decoding Schema {decoding_schema}, Task {task_name} do not map to constraint decoder.'
        )
    return task_map[task_name](tokenizer=tokenizer, type_schema=type_schema, source_prefix=source_prefix)
