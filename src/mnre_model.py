
import torch
import torch.nn as nn

from transformers import T5ForConditionalGeneration
from modeling_t5 import VLT5

class VLT5MNRE(VLT5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):
        device = next(self.parameters()).device

        batch = self.vis_forward(batch, device)
        
        # task = batch["task"]
        # print(type(batch['vis_feats']))
        
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        
        # decoder_input_ids = batch['decoder_input_ids'].to(device)
        # generated_sents = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # print(generated_sents)
        lm_labels = batch["labels"].to(device)
        # word_mask = lm_labels != -100
        # lm_labels[~word_mask] = self.tokenizer.pad_token_id
        # generated_sents = self.tokenizer.batch_decode(lm_labels, skip_special_tokens=False,clean_up_tokenization_spaces=False)
      
        # print(generated_sents)
        
        reduce_loss = True
        output = self(
            input_ids=input_ids,
            attention_mask = attention_mask,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            decoder_input_ids = decoder_input_ids,
            reduce_loss=reduce_loss,
            # task=None,
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device

        batch = self.vis_forward(batch, device)
        
        vis_feats = batch['vis_feats'].to(device)
        vis_pos = batch['boxes'].to(device)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)

        # def prefix_allowed_tokens_fn(batch_id, sent):
        #     src_sentence = input_ids[batch_id]
        #     return constraint_decoder.constraint_decoding(src_sentence=src_sentence, tgt_generated=sent)
        
        # gen_kwargs = {}
        # gen_kwargs['num_beams'] = 5
        # gen_kwargs['max_length'] = 192
        # kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

        # print(kwargs)
        
        output = self.generate(
            input_ids=input_ids,
            # attention_mask = attention_mask,
            vis_inputs=(vis_feats, vis_pos),
            # task=None,
            **kwargs,
        )

        # generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=False,clean_up_tokenization_spaces=False)
        # print(generated_sents)
        
        result = {}
        result['pred'] = output



        return result

