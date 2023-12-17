
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
# from transformer import T5ForConditionalGeneration
from modeling_t5 import VLT5
class UMIEModel(VLT5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):
        device = next(self.parameters()).device

        batch = self.vis_forward(batch, device)
        
        task = batch["task"]
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
            task=task
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss1 = output['loss']
        # if True:
        #     word_mask = lm_labels != -100
        #     lm_labels_input = copy.deepcopy(lm_labels)
        #     lm_labels_input[~word_mask] = self.tokenizer.pad_token_id
        #     output = self(
        #         input_ids=lm_labels_input,
        #         attention_mask = ~word_mask,
        #         vis_inputs=(vis_feats, vis_pos),
        #         labels=lm_labels,
        #         decoder_input_ids = decoder_input_ids,
        #         reduce_loss=reduce_loss,
        #     )
            
        #     loss2 = output['loss']
            
        # vis_embedding = output['vis_encoder_last_hidden_state']
        # txt_embedding = output['encoder_last_hidden_state']
        # # mm_embedding = output["attn_output"]
        # T = torch.tensor(0.5)

        # # embeddings_t = txt_embedding[: , -1 , :]
        # # embeddings_v = txt_embedding[: , 0  , :]


        # # embeddings_m = mm_embedding[: , 0, : ]
        # embeddings_t = self.proj_t(embeddings_t)
        # embeddings_t = self.proj_t_2(embeddings_t)
        # embeddings_v = self.proj_v(embeddings_v)
        # embeddings_v = self.proj_v_2(embeddings_v)

        # embeddings_t = torch.mean(embeddings_t, dim = 1)
        # embeddings_v = torch.mean(embeddings_v, dim = 1)
        # # embedding_m = self.prog_v(embeddings_m)
        # embeddings_t = F.normalize(embeddings_t, p=2, dim=1)
        # embedding_m = F.normalize(embeddings_v, p=2, dim=1)

        # # print(embeddings_t.shape)
        # # print(embeddings_v.shape)
        # logits = (embeddings_t @ embeddings_v.T) * torch.exp(T)
        # # logits /= T
        # target = torch.arange(len(vis_feats), device=device)
        # # print(target.shape,logits.shape)
        # loss_t = F.cross_entropy(logits, target)
        # loss_v = F.cross_entropy(logits.T, target)
        # CL_loss = (loss_t + loss_v) / 2
        CL_loss = 0
        result = {
            'loss': loss1  + CL_loss
        }
        return result
        
    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device

        batch = self.vis_forward(batch, device)
        
        vis_feats = batch['vis_feats'].to(device)
        vis_pos = batch['boxes'].to(device)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        task = batch["task"]

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
            task=task,
            **kwargs,
        )
        # print(output)

        # generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=False,clean_up_tokenization_spaces=False)
        # print(generated_sents)
        
        result = {}
        result['pred'] = output



        return result


