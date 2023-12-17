from dataclasses import dataclass
import pdb
from my_transformers.modeling_t5 import (
    T5Stack, T5Block, T5LayerNorm, T5LayerFF, T5LayerCrossAttention,
    T5ForConditionalGeneration, T5GateAttention
)
from transformers.activations import ACT2FN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from typing import Any, Dict, List, Optional, Tuple
import copy
from clip.clip import load

from transformers.modeling_outputs import ModelOutput, BaseModelOutput,  BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging

from sample import Downsample, SparseSample, OneDDownsample
from timm.models.vision_transformer import resize_pos_embed

from adapters import (
    AdapterLayer, 
    AdapterController,
    OutputParallelAdapterLayer, 
    TaskEmbeddingController,
    AdapterLayersHyperNetController,
    AdapterLayersOneHyperNetController,
    MetaLayersAdapterController
)

from adapters.hypercomplex.layers import PHMLinear

from prompt import (
    PromptController,
)

# from utils import *

logger = logging.get_logger(__name__)



class GateBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.d_model = config.d_model
        self.layer = nn.ModuleList()
        self.layer.append(T5GateAttention(config))
        self.layer.append(T5LayerFF(config))
        self.feat = nn.Linear(512, self.d_model, bias=False)
        self.proj_t = nn.Linear(self.d_model, 384, bias=False)
        self.proj_t_2 = nn.Linear(384, 128, bias=False)

        self.proj_v = nn.Linear(self.d_model, 384, bias=False)
        self.proj_v_2 = nn.Linear(384, 128, bias=False)
 
    def forward(self,
                vis_feat,
                q_text,
                k_img,
                v_img,
                att_mask,
                task=None
            ):
        
        cross_attention_outputs = self.layer[0](
                q_text,
                key_value_states=k_img,
                task=task,
        )
        hidden_states = cross_attention_outputs[0]
        attn_output = self.layer[-1](hidden_states , None, task= task)
        ##update clip
        # vis_feat = self.feat(vis_feat)

        # import pdb
        # pdb.set_trace()


        import torch.nn.functional as F

        embeddings_t = self.proj_t(q_text)
        embeddings_t = self.proj_t_2(embeddings_t)
        embeddings_v = self.proj_v(k_img)
        embeddings_v = self.proj_v_2(embeddings_v)

        
        
        embeddings_t = torch.mean(embeddings_t, dim = 1)
        embeddings_v = torch.mean(embeddings_v, dim = 1)
        # print(embeddings_t)
        # print(embeddings_v)
        embeddings_t = F.normalize(embeddings_t, p=2, dim = 1)
        embeddings_v = F.normalize(embeddings_v, p=2, dim = 1)
        # rate = torch.cosine_similarity(embeddings_v, embeddings_t, dim=1, eps=1e-08)
        # print(rate)
        
        T = torch.tensor(0.5)
        logits = (embeddings_t @ embeddings_v.T) * torch.exp(T)
        
        rate = F.leaky_relu(torch.diag(logits))  
        # rate = torch.tensor(0.).to(rate.device)
        gate = q_text + F.leaky_relu(rate.view(-1,1,1))* attn_output

        CL_loss = 0

        return gate, attn_output, CL_loss
        
class VStack(T5Stack):
    def __init__(self, config, embed_tokens=None, task_embed=None):
       
        super().__init__(config)   
        self.embed_tokens = embed_tokens
        self.task_embed = task_embed
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(2)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.init_weights()
        self.model_parallel = False
        self.device_map = None
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,
    ):
        # Model parallel


        use_cache = use_cache if use_cache is not None else self.config.use_cache #True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions #False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length



        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)


        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)


        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)



       
        if past_key_values is None:
            past_key_values = [None] * len(self.block)
        
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            block_adapters = None


            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                block_adapters=block_adapters,
                task=task,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[2]

            # append next layer key value states
      

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
        


        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class VisualEmbedding(nn.Module):
    def __init__(self, config, obj_order_embedding):
        super().__init__()
        self.config = config
        feat_dim = 512
        pos_dim = config.pos_dim
        # n_objs = config.n_objs
        n_images = config.n_images
        self.vstack = VStack(config)
        if self.config.individual_vis_layer_norm:

            # Object feature encoding
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]

            if self.config.use_vis_layer_norm:
                feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))

            for i in range(config.additional_visual_embedding_layers):
                feat_embedding.append(nn.Linear(config.d_model, config.d_model))
                feat_embedding.append(nn.ReLU(True))

                if self.config.use_vis_layer_norm:
                    feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            
            self.feat_embedding = nn.Sequential(*feat_embedding)

            # self.relative_vis_pos_embedding = nn.Linear(pos_dim + 1, config.num_heads)
            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            if self.config.use_vis_layer_norm:
                absolute_vis_pos_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)
            # self.absolute_vis_pos_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

            if self.config.use_vis_order_embedding:
                # self.obj_order_embedding = nn.Embedding(n_objs, config.d_model)
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)

        else:
            # Object feature encoding
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            # if self.config.use_vis_layer_norm:
            #     feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))

            for i in range(config.additional_visual_embedding_layers): #0
                feat_embedding.append(nn.Linear(config.d_model, config.d_model))
                feat_embedding.append(nn.ReLU(True))

            self.feat_embedding = nn.Sequential(*feat_embedding)

            # self.relative_vis_pos_embedding = nn.Linear(pos_dim + 1, config.num_heads)
            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            # if self.config.use_vis_layer_norm:
            #     absolute_vis_pos_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)
            # self.absolute_vis_pos_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

            if self.config.use_vis_order_embedding:
                # self.obj_order_embedding = nn.Embedding(n_objs, config.d_model)
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)

            if self.config.use_vis_layer_norm:
                self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def get_area(self, pos):
        """
        Args
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            area : [B, N]
        """
        # [B, N]
        ##modify
        height = pos[:, :, 3] - pos[:, :, 1]
        width = pos[:, :, 2] - pos[:, :, 0]
        area = height * width
        return area


    def forward(self, feats, pos, task=None, img_order_ids=None, obj_order_ids=None):
        """
        Args
            feats: [B, N, feat_dim]
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            relative_vis_pos_embedding: [B, N, N, n_heads]
            absolute_vis_pos_embedding: # [B, N, d_model]
        """
  
        
        B, N, _ = feats.size()
        assert pos.size() == (B, N, 4)
        
        # import pdb 
        # pdb.set_trace()
        
        feat_embedding = self.feat_embedding(feats)

        device = feats.device
        dtype = feats.dtype

        area = self.get_area(pos).unsqueeze(2) # [B, N, 1]
        # print(area)
        pos = torch.cat([pos, area], dim=2) # [B, N, 5]
        
        # [B, N, d_model]
        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos) #torch.Size([32, 36, 768])


        if self.config.use_vis_order_embedding: #True
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0) #.expand(B, -1)
            img_order_embedding = self.img_order_embedding(img_order_ids)

            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0) #.expand(B,-1)
            # assert obj_order_ids.max().item() < 32200, obj_order_ids
            obj_order_ids = self.obj_order_embedding.num_embeddings - obj_order_ids - 1
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)

            vis_embedding = feat_embedding + absolute_vis_pos_embedding + \
                img_order_embedding + obj_order_embedding #torch.Size([32, 36, 768])

        else:
            vis_embedding = feat_embedding + absolute_vis_pos_embedding 

        if not self.config.individual_vis_layer_norm:
            if self.config.use_vis_layer_norm:
                vis_embedding = self.layer_norm(vis_embedding)
        

        vis_embedding = self.vstack(inputs_embeds = vis_embedding, task=task)
        vis_embedding = vis_embedding[0]
        
        return vis_embedding
    
class ClipVision(nn.Module):    
    def __init__(self,image_size : int = 224):
        super(ClipVision,self).__init__()
        self.model, self.transform = load("RN101", jit=False)
        self.num_patches = (image_size // 32) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.model.visual.attnpool.positional_embedding.shape[-1]),)
        self.pos_embed.weight = resize_pos_embed(self.model.visual.attnpool.positional_embedding.unsqueeze(0), self.pos_embed)
        self.model.visual.attnpool.positional_embedding = self.pos_embed
    
    def forward(self,image):
        B = image.shape[0]
        att, fc = self.model.encode_image(image)
        att = att.permute(0, 2, 3, 1).to(att.dtype)
        att = att.reshape(B, 7 ** 2, -1)
        return fc, att
         
    

class JointEncoder(T5Stack):
    def __init__(self, config, embed_tokens=None, task_embed=None):
        super().__init__(config, embed_tokens, task_embed)
        self.config = config

        assert self.config.is_decoder is False
        
        # self.vis_encoder = ClipVision()
        self.visual_embedding = VisualEmbedding(self.config, embed_tokens)
        self.gate  = GateBlock(self.config)
        self.downsample = None
        self.sparse_sample = None
        if config.oneddownsample:
            self.downsample = OneDDownsample(config.n_boxes)
        elif config.downsample:
            sqrt_size = int(config.n_boxes ** 0.5)
            output_size = (sqrt_size, sqrt_size)
            self.downsample = Downsample(output_size)
        elif config.sparse_sample:
            self.sparse_sample = SparseSample(config.n_boxes)

        if config.encoder_prompt_config:
            self.prompt_modules = PromptController(config.encoder_prompt_config)
        else:
            self.prompt_modules = None

        self.init_weights()

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
        self.visual_embedding.obj_order_embedding = new_embeddings

    def get_prompt(self, bsz, device):
        input_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        return self.prefix_embedding(input_tokens)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,
    ):

        
        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        if self.prompt_modules is not None:
            prefix_embeds = self.prompt_modules(inputs_embeds.shape[0], inputs_embeds.device, task)
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        B, L = inputs_embeds.size()[:-1]
        
        
        ##update clip
        # vis_input_fc,vis_input_arr = self.vis_encoder(vis_inputs[0])
        # vis_input = (vis_input_arr, vis_inputs[1])
    
        
        # import pdb
        # pdb.set_trace()

        if self.downsample is not None:
            vis_inputs = self.downsample(vis_input)
        
        # import pdb
        # pdb.set_trace()
        vis_feats = vis_inputs[0].to(inputs_embeds.dtype)
        boxes = vis_inputs[1].to(inputs_embeds.dtype)
        img_order_ids = None
        obj_order_ids = None
        if len(vis_inputs) >= 3:
            img_order_ids = vis_inputs[2]
        if len(vis_inputs) == 4:
            obj_order_ids = vis_inputs[3]

        vis_embeds = self.visual_embedding(
            vis_feats, boxes, task, img_order_ids, obj_order_ids)

        if self.sparse_sample is not None:
            vis_embeds = self.sparse_sample(vis_embeds)

        V_L = vis_embeds.size(1)
        #not 
        # vis_input_fc = vis_embeds[:,0,:]

        ## 
        # inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        if vis_attention_mask is None:
            vis_attention_mask = attention_mask.new_ones(B, V_L)

        if self.prompt_modules is not None:
            prefix_attention_mask = torch.ones(
                B, prefix_embeds.shape[1], dtype=inputs_embeds.dtype, device=inputs_embeds.device
            )

            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        # attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask = self.get_extended_attention_mask(
        #     attention_mask,
        #     (B, L+V_L),
        #     inputs_embeds.device)
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            (B, L),
            inputs_embeds.device)
        
        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        # position_bias = None
        # encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        if self.config.num_layers > 0:

            assert self.block[0].layer[0].SelfAttention.has_relative_attention_bias

            # seq_length = L + V_L
            seq_length = L
            q_len = seq_length
            k_len = seq_length

            # [1, n_heads, Q_len, K_len]
            text_position_bias = self.block[0].layer[0].SelfAttention.compute_bias(
                L, L)
            num_heads = text_position_bias.size(1)
            position_bias = text_position_bias.new_zeros(
                1, num_heads, seq_length, seq_length)
            position_bias[:, :, :L, :L] = text_position_bias

            # print('position_bias size', position_bias.size())
            # print('attention_mask size', attention_mask.size())
            # print('extended_attention_mask size', extended_attention_mask.size())
            # relative position bias only between Text <-> Text
            # no relative position bias Text -> Vision
            # no relative position bias Vision -> Text
            # no relative position bias Vision <-> Vision
            # position_bias[:, :, L:, :] = 0
            # position_bias[:, :, :, L:] = 0
            position_bias = position_bias + extended_attention_mask

            task_embedding = None
            if task is not None and self.task_embed is not None:
                task_embedding = self.task_embed(task)

            for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):

                # if output_hidden_states:
                #     all_hidden_states = all_hidden_states + (hidden_states,)

                block_adapters = None
                if self.adapter_layers_hyper_net:
                    block_adapters = self.adapter_layers_hyper_net(task_embedding, i)

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    encoder_decoder_position_bias=None,
                    head_mask=head_mask[i],
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    block_adapters=block_adapters,
                    task=task,
                )
                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                hidden_states, present_key_value_state = layer_outputs[:2]

                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights),
                # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[2]

                # append next layer key value states
                if use_cache:
                    present_key_value_states = present_key_value_states + \
                        (present_key_value_state,)

                # if output_attentions:
                #     all_attentions = all_attentions + (layer_outputs[3],)
                #     if self.is_decoder:
                #         all_cross_attentions = all_cross_attentions + \
                #             (layer_outputs[5],)
                
        extended_vis_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            (B, V_L),
            inputs_embeds.device).unsqueeze(dim=1)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        attention_mask = torch.cat([vis_attention_mask, attention_mask], dim=1)
        
        #region
        vis_input_fc = vis_embeds[:, 0, :]
        hidden_states, attn_output, CL_loss = self.gate(vis_input_fc, hidden_states, vis_embeds, vis_embeds, extended_vis_attention_mask, task)
        # hidden_states = torch.cat([vis_embeds, hidden_states], dim=1)
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    vis_embeds,
                    attn_output,
                    CL_loss,
                ]
                if v is not None
            )
    
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class VLT5(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        # super(T5ForConditionalGeneration, self).__init__(config)
        super().__init__(config)

        self.config = config

        self.d_model= config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        #---- Modified ----#
        # self.encoder = T5Stack(encoder_config, self.shared)
        
        self.encoder = JointEncoder(encoder_config, self.shared, self.shared_task_embed)
        #------------------#

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers

        self.proj_t = nn.Linear(config.d_model, 384, bias=False)
        self.proj_t_2 = nn.Linear(384, 128, bias=False)

        self.proj_v = nn.Linear(config.d_model, 384, bias=False)
        self.proj_v_2 = nn.Linear(384, 128, bias=False)

        self.decoder = T5Stack(decoder_config, self.shared, self.shared_task_embed)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.decoder_prompt_config:
            self.prompt_modules = PromptController(config.decoder_prompt_config)
        else:
            self.prompt_modules = None

        if config.use_lm_head_adapter:
            self.output_adapter = OutputParallelAdapterLayer(config, self.model.shared.num_embeddings)

        adapter_config = config.adapter_config

        if adapter_config is not None:
            if getattr(adapter_config, "train_task_adapters", None) and getattr(adapter_config, "hypercomplex_adapters", None):
                if adapter_config.shared_phm_rule:
                    phm_dim = adapter_config.hypercomplex_division
                    self.factorized_phm_rule = adapter_config.factorized_phm_rule
                    if self.factorized_phm_rule:
                        self.phm_rule_left = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, 1),
                            requires_grad=adapter_config.learn_phm)
                        self.phm_rule_right = nn.Parameter(torch.FloatTensor(phm_dim, 1, phm_dim),
                            requires_grad=adapter_config.learn_phm)
                        if adapter_config.phm_c_init == "normal":
                            self.phm_rule_left.data.normal_(mean=0, std=adapter_config.phm_init_range)
                            self.phm_rule_right.data.normal_(mean=0, std=adapter_config.phm_init_range)
                        elif adapter_config.phm_c_init == "uniform":
                            self.phm_rule_left.data.uniform_(-1, 1)
                            self.phm_rule_right.data.uniform_(-1, 1)
                        else:
                            raise NotImplementedError
                    else:
                        self.phm_rule = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, phm_dim),\
                            requires_grad=adapter_config.learn_phm)
                        if adapter_config.phm_c_init == "normal":
                            self.phm_rule.data.normal_(mean=0, std=adapter_config.phm_init_range)
                        elif adapter_config.phm_c_init == "uniform":
                            self.phm_rule.data.uniform_(-1, 1)
                        else:
                            raise NotImplementedError 
                    self.set_phm_rule()

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def set_phm_rule(self):
        def set_phm_rule(module):
            # TODO: we need to check there is one of these, and this is activated.
            for name, sub_module in module.named_modules():
                if isinstance(sub_module, PHMLinear):
                    if self.factorized_phm_rule:
                        sub_module.set_phm_rule(phm_rule_right=self.phm_rule_right, 
                                                phm_rule_left=self.phm_rule_left)
                    else:
                        sub_module.set_phm_rule(phm_rule=self.phm_rule)
        set_phm_rule(self.encoder)
        set_phm_rule(self.decoder)

    def get_prompt(self, bsz, device):
        input_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(device) # (B, L)
        prefix_prompt = self.prefix_embedding(input_tokens) # (B, L, d_model)

        temp_results = self.decoder(inputs_embeds=prefix_prompt, use_cache=True, return_dict=True)

        past_key_values = temp_results.past_key_values

        # past_key_values = list(past_key_values)

        # for layer in range(len(past_key_values)):
        #     past_key_values[layer] = list(past_key_values[layer])
        #     past_key_values[layer].append(None)
        #     past_key_values[layer].append(None)
        
        return past_key_values

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def extend_vocab(self, vocab_size):

        new_shared = nn.Embedding(vocab_size, self.config.d_model)
        old_weight = self.shared.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_shared.weight.data[:old_vocab_size, :] = old_weight
        self.shared = new_shared

        new_lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        old_weight = self.lm_head.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_lm_head.weight.data[:old_vocab_size, :] = old_weight
        self.lm_head = new_lm_head

        self.vis_encoder.visual_embedding.obj_order_embedding = self.shared

        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        self.lm_head.weight = self.shared.weight

        self.config.vocab_size = vocab_size
        self.encoder.config.vocab_size = vocab_size
        self.vis_encoder.config.vocab_size = vocab_size
        self.decoder.config.vocab_size = vocab_size


    # @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,

        vis_inputs=None,
        vis_attention_mask=None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,
        task=None,
        **kwargs,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,

                vis_inputs=vis_inputs,
                vis_attention_mask=vis_attention_mask,

                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                task=task
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )
        # import pdb
        # pdb.set_trace()
        hidden_states = encoder_outputs[0]

        # torch.Size([16, 86, 1024])
        # 50 text
        # 36 vis
        # import pdb
        # pdb.set_trace()

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=hidden_states.dtype, device=hidden_states.device)

        if self.config.encoder_prompt_config is not None and self.config.encoder_prompt_config.prompt_len > 0:
            prefix_attention_mask = torch.ones(
                attention_mask.shape[0], 
                self.config.encoder_prompt_config.prompt_len, 
                dtype=attention_mask.dtype, 
                device=attention_mask.device,
            )

            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        
        if vis_attention_mask is None:
            B, L = attention_mask.size()
            V_L = encoder_outputs[0].size(1) - L
            vis_attention_mask = attention_mask.new_ones(B, V_L)
        encoder_attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        if self.prompt_modules is not None and past_key_values is None:
            prefix_embeds = self.prompt_modules(B, attention_mask.device, task)

            past_key_values = self.decoder(inputs_embeds=prefix_embeds, use_cache=True, return_dict=True).past_key_values

        
        # Decode
        # import pdb
        # pdb.set_trace()
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,

            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=None,
            task=task,
        )
        # print('decoder_outputs')
        # print(decoder_outputs)

        sequence_output = decoder_outputs[0]
        # print(sequence_output == decoder_outputs.last_hidden_state)
        # assert self.config.tie_word_embeddings is True
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        if return_hidden_state:
            return sequence_output

        lm_logits = self.lm_head(sequence_output)

        loss = None
        CL_loss = None
        vis_output = None
        attn_output = None
        if labels is not None:
           
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            # loss = loss_fct(
            #     lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            # print(x.shape,y.shape)
            # import pdb
            # pdb.set_trace()
            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
                
            # import pdb
            # pdb.set_tace()
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1))   
            # + 0.01 * CL_loss + 0.02 * CL_loss_1

            # print('loss')
            # print(loss)

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        return VLSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            vis_encoder_last_hidden_state = vis_output,
            # decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=hidden_states,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            # vis_encoder_last_hidden_state=vis_encoder_outputs.last_hidden_state,
            # vis_encoder_hidden_states=vis_encoder_outputs.hidden_states,
            # vis_encoder_attentions=vis_encoder_outputs.attentions,
            # cross_encoder_outputs=cross_encoder_outputs
        )

    def vis_forward(self, batch, device):
        if hasattr(self, "vis_encoder"):
            # self.vis_encoder.eval() # freeze the batchnorm statistics
            images = batch["images"].to(device)

            if self.config.vis_pooling_output: #False
                _, vis_feats = self.vis_encoder(images)
            else:
                vis_feats, _ = self.vis_encoder(images)
            # vis_feats: (B, dim, L ** 0.5, L ** 0.5)
            B, L, D = vis_feats.shape
            vis_pos = torch.zeros(B, L, 4, dtype=vis_feats.dtype)

            batch["vis_feats"] = vis_feats
            batch["boxes"] = vis_pos

        return batch

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None,
        encoder_outputs=None,
        **kwargs):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        output = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

        if 'vis_attention_mask' in kwargs:
            output['vis_attention_mask'] = kwargs['vis_attention_mask']

        if "task" in kwargs:
            output["task"] = kwargs["task"]

        return output
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)
    
    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1,
                                                                expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)

        if model_kwargs.get("vis_attention_mask", None) is not None:
            model_kwargs['vis_attention_mask'] = model_kwargs['vis_attention_mask'].index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs


@dataclass
class VLSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Languaged modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see ``past_key_values`` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    vis_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    vis_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vis_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    # cross_encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None


if __name__ == "__main__":
    import transformers
    import re

    from adapters import MetaAdapterConfig, AdapterConfig, CompactorConfig

    config = transformers.AutoConfig.from_pretrained("t5-base")

    config.feat_dim = 2048
    config.pos_dim = 4
    config.n_images = 2

    config.use_vis_order_embedding = True
    config.additional_visual_embedding_layers = 0
    config.preseqlen = 0
    config.decoder_preseqlen = 0

    config.dropout_rate = 0.1
    config.dropout = 0.1
    config.attention_dropout = 0.1
    config.activation_dropout = 0.1

    config.use_vis_layer_norm = True
    config.individual_vis_layer_norm = True
    config.losses = 'lm,obj,attr,feat'
    config.tasks = "vqa,gqa"

    config.share_vis_lang_layer_norm = False
    config.classifier = False

    config.downsample = False
    config.sparse_sample = False

    config.add_layer_norm_before_adapter = True
    config.add_layer_norm_after_adapter = True

    config.use_lm_head_adapter = False

    config.use_hyperformer = False
    config.use_adapter = False
    config.use_compacter = True

    if config.use_hyperformer or config.use_adapter or config.use_compacter:

        assert config.use_hyperformer + config.use_adapter + config.use_compacter <= 1, "You can only at most one kind of adapters."
        if config.use_hyperformer:
            CONFIG_CLASS = MetaAdapterConfig
        elif config.use_adapter:
            CONFIG_CLASS = AdapterConfig
        elif config.use_compacter:
            CONFIG_CLASS = CompactorConfig

        config.adapter_config = CONFIG_CLASS()
        config.adapter_config.tasks = re.split("[, ]+", config.tasks) # tranform to list
        config.adapter_config.input_dim = 768
        config.adapter_config.d_model = 768
        config.adapter_config.use_single_adapter = True

    else:
        config.adapter_config = None

    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")

    num_added_toks = 0
    additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
            [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
    special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    config.default_obj_order_ids = tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

    model = VLT5.from_pretrained("t5-base", config=config)
    model.resize_token_embeddings(tokenizer.vocab_size)
    model.tokenizer = tokenizer

    inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
    
    vis_feats = torch.randn(1, 36, 2048)
    vis_pos = torch.randn(1, 36, 4)

    generation_output = model.generate(
                **inputs,
                vis_inputs=(vis_feats, vis_pos),
                task="gqa"
    )
    
    # print(generation_output)

    # print(tokenizer.batch_decode(generation_output, skip_special_tokens=True))


    orig_param_size = 222903552

    adapter_param = 0
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, (AdapterController, TaskEmbeddingController, AdapterLayersHyperNetController, AdapterLayersOneHyperNetController, MetaLayersAdapterController)):
            print(f"{name} is trainable...")
            for param_name, param in sub_module.named_parameters():
                adapter_param += param.numel()

    visual_param = 0
    targets = ["visual_embedding", "prefix_embedding"]
    # unfreeze the parameters in targets anyway
    for n, p in model.named_parameters():
        if any(t in n for t in targets):
            visual_param += p.numel()
            print(f"{n} is trainable...")

    print(f"adapter params: {adapter_param}, {adapter_param / orig_param_size * 100:.3f} %")

    print(f"visual params: {visual_param}, {visual_param / orig_param_size * 100:.3f} %")

    total_param = adapter_param + visual_param
    print(f"total params: {total_param}, {total_param / orig_param_size * 100:.3f} %")
