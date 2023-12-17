
import torch
import torch.nn as nn

# from transformer import T5ForConditionalGeneration
from modeling_t5 import VLT5
class VLT5SWiG(VLT5):
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
        vis_attention_mask = batch['vis_attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        
        # decoder_input_ids = batch['decoder_input_ids'].to(device)
        # generated_sents = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # print(generated_sents)
        # import pdb
        # pdb.set_trace()
        lm_labels = batch["labels"].to(device)
        
        
        reduce_loss = True
        output = self(
            input_ids=input_ids,
            attention_mask = attention_mask,
            vis_inputs=(vis_feats, vis_pos),
            # vis_attention_mask = vis_attention_mask,
            labels=lm_labels,
            decoder_input_ids = decoder_input_ids,
            reduce_loss=reduce_loss,
            # task=None,
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']
        # vis_embedding = output['vis_encoder_last_hidden_state']
        # txt_embedding = output['encoder_last_hidden_state']
        # T = 1.0 
        # # import pdb
        # # pdb.set_trace()
        # import torch.nn.functional as F

        # embeddings_t = txt_embedding[:,-1,:]
        # embeddings_v = vis_embedding[:,0,:]
        # embeddings_t = model.proj_t(embeddings_t)
        # embeddings_v = model.proj_v(embeddings_v)
        # embeddings_t = F.normalize(embeddings_t, p=2, dim=1)
        # embeddings_v = F.normalize(embeddings_v, p=2, dim=1)
        # # print(embeddings_t.shape)
        # # print(embeddings_v.shape)
        # logits = (embeddings_t @ embeddings_v.T)
        # logits /= T
        # target = torch.arange(len(vis_feats), device=device)
        # # print(target.shape,logits.shape)
        # loss_t = F.cross_entropy(logits, target)
        # loss_v = F.cross_entropy(logits.T, target)
        # # if task == "swig_event":
        # #     import pdb
        # #     pdb.set_trace()
        # CL_loss = (loss_t + loss_v) / 2

        # result = {
        #     'loss':  CL_loss
        # }
        # if task == "ace05":
            # result = {
            #     'loss':output["loss"]
            # }
            # return result
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


        
        gen_kwargs = {}
        gen_kwargs['num_beams'] = 1
        gen_kwargs['max_length'] = 192


        output = self.generate(
            input_ids=input_ids,
            # attention_mask = attention_mask,
            vis_inputs=(vis_feats, vis_pos),
            # task=None,
            **gen_kwargs,
        )

        # generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=False,clean_up_tokenization_spaces=False)
        # print(generated_sents)
        
        result = {}
        result['pred'] = output



        return result

