import os
import re
from pathlib import Path
from packaging import version

import torch
import torch.nn as nn
import logging
from pprint import pprint

from utils import load_state_dict, LossMeter, set_global_logging_level, FusedOptimizer
from transformers.models.t5.modeling_t5 import T5LayerNorm
import modeling_t5
from modeling_t5 import VStack,GateBlock
from adapters import (
    AdapterController,
    OutputParallelAdapterLayer,
    MetaAdapterConfig,
    AdapterConfig,
    CompactorConfig,
    LRAdapterConfig,
    TaskEmbeddingController,
    AdapterLayersHyperNetController,
    AdapterLayersOneHyperNetController
)
from transformers import (
    AutoTokenizer
)
from prompt import EncoderPromptConfig, DecoderPromptConfig, PromptController
from lora import LoraConfig

from vis_encoder import CLIPResNetEncoder
from clip.model import VisualAdapter



proj_dir = Path(__file__).resolve().parent.parent

_use_native_amp = False
_use_apex = False
use_deepspeed = False

uie = False
# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


class TrainerBase(object):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.args = args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.verbose = True
        if self.args.distributed or self.args.deepspeed:
            if self.args.gpu != 0:
                self.verbose = False

        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        self.include_vis_encoder = self.args.feature_type.startswith("raw")
        # self.deepspeed = args.deepspeed

    def create_config(self):
        from transformers import T5Config, BartConfig

        if 't5' in self.args.backbone:
            config_class = T5Config
        elif 'umie' in self.args.backbone:
            config_class = T5Config
        elif 'bart' in self.args.backbone:
            config_class = BartConfig
        else:
            return None

        # if 'uie' in self.args.backbone:
        #     config = config_class.from_pretrained("/home/liqingyuan/VL_adapter/VL-T5/hf_models/uie-base-en")
        #     # config = AutoConfig.from_pretrained(
        #     #     "/home/liqingyuan/VL_adapter/VL-T5/hf_models/uie-base-en",
        #     #     cache_dir=None,
        #     #     revision="main",
        #     #     use_auth_token=None,
        #     #     )
        #     # return config
        # else:
        config = config_class.from_pretrained(self.args.backbone)
        args = self.args


        for k, v in vars(args).items():
            setattr(config, k, v)

        config.feat_dim = args.feat_dim
        config.pos_dim = args.pos_dim
        config.n_images = 2
        config.n_boxes = args.n_boxes
        config.expand_vis_embedding = args.expand_vis_embedding
        config.n_image_tokens = args.n_image_tokens
        config.vis_use_transformer = args.vis_use_transformer
        config.downsample = args.downsample
        config.oneddownsample = args.oneddownsample
        config.sparse_sample = args.sparse_sample
        # config.downsample = False
        # config.oneddownsample = False
        # config.sparse_sample = False
        config.use_vis_order_embedding = args.use_vis_order_embedding
        config.additional_visual_embedding_layers = args.additional_visual_embedding_layers
        config.mid_dim = args.mid_dim
        config.reduction_factor = args.reduction_factor
        config.vstack = False
        config.vis_pooling_output = args.vis_pooling_output

        config.use_lm_head_adapter = args.use_lm_head_adapter

        config.use_hyperformer = args.use_hyperformer
        config.use_adapter = args.use_adapter
        config.use_compacter = args.use_compacter
        config.use_lradapter = args.use_lradapter

        config.add_adapter_cross_attn = args.add_adapter_cross_attn

        tasks = re.split("[, ]+", args.tasks) # tranform to list

        if args.use_hyperformer or args.use_adapter or args.use_compacter or args.use_lradapter:
            
            assert config.use_hyperformer + config.use_adapter + config.use_compacter + config.use_lradapter <= 1, "You can only at most one kind of adapters."

            if args.use_hyperformer:
                CONFIG_CLASS = MetaAdapterConfig
            elif args.use_adapter:
                CONFIG_CLASS = AdapterConfig
            elif args.use_compacter:
                CONFIG_CLASS = CompactorConfig
            elif args.use_lradapter:
                CONFIG_CLASS = LRAdapterConfig

            config.adapter_config = CONFIG_CLASS()
            config.adapter_config.tasks = tasks
            config.adapter_config.input_dim = config.d_model # for hyperformer
            config.adapter_config.d_model = config.d_model # for adapter and compactor
            config.adapter_config.unique_hyper_net = args.unique_hyper_net
            config.adapter_config.efficient_unique_hyper_net = args.efficient_unique_hyper_net
            config.adapter_config.use_single_adapter = args.use_single_adapter
            config.adapter_config.hypercomplex_division = args.hypercomplex_division
            config.adapter_config.phm_rank = args.phm_rank
            config.adapter_config.shared_phm_rule = args.shared_phm_rule
            config.adapter_config.factorized_phm = args.factorized_phm
            config.adapter_config.low_rank_rank = args.low_rank_rank
            config.adapter_config.phm_init_range = args.phm_init_range

            config.adapter_config.share_down_sampler = args.share_down_sampler
            config.adapter_config.share_up_sampler = args.share_up_sampler
            config.adapter_config.reduction_factor = args.reduction_factor
            config.adapter_config.shared_phm_rule_over_tasks = args.shared_phm_rule_over_tasks

            config.adapter_config.add_layer_norm_before_adapter = args.add_layer_norm_before_adapter
            config.adapter_config.add_layer_norm_after_adapter = args.add_layer_norm_after_adapter

            config.adapter_config.track_z = args.track_z

            if args.projected_task_embedding_dim != -1:
                config.adapter_config.projected_task_embedding_dim = args.projected_task_embedding_dim
        else:
            config.adapter_config = None

        # for prompt        
        if args.encoder_prompt_len > 0:
            config.encoder_prompt_config = EncoderPromptConfig()
            config.encoder_prompt_config.prompt_len = args.encoder_prompt_len
            config.encoder_prompt_config.tasks = tasks
            config.encoder_prompt_config.use_single_prompt = args.use_single_prompt
            config.encoder_prompt_config.mid_dim = args.mid_dim
        else:
            config.encoder_prompt_config = None

        if args.decoder_prompt_len > 0:
            config.decoder_prompt_config = DecoderPromptConfig()
            config.decoder_prompt_config.prompt_len = args.decoder_prompt_len
            config.decoder_prompt_config.tasks = tasks
            config.decoder_prompt_config.use_single_prompt = args.use_single_prompt
            config.decoder_prompt_config.mid_dim = args.mid_dim
        else:
            config.decoder_prompt_config = None

        # for lora
        if args.use_lora:
            config.lora_config = LoraConfig()
            config.lora_config.lora_dim = args.lora_dim
            config.lora_config.lora_alpha = args.lora_alpha
            config.lora_config.tasks = tasks
            config.lora_config.use_single_lora = args.use_single_lora
        
        config.dropout_rate = args.dropout
        config.dropout = args.dropout
        config.attention_dropout = args.dropout
        config.activation_dropout = args.dropout

        config.use_vis_layer_norm = args.use_vis_layer_norm
        config.individual_vis_layer_norm = args.individual_vis_layer_norm
        config.losses = args.losses

        config.share_vis_lang_layer_norm = args.share_vis_lang_layer_norm
        config.classifier = args.classifier

        return config

    def create_model(self, model_class, config=None, **kwargs):
        print(f'Building Model at GPU {self.args.gpu}')

        model_name = self.args.backbone


        # if self.args.uie and 't5' in self.args.backbone:
        #     print("args.uie")
        #     model = model_class.from_pretrained(
        #     "/home/liqingyuan/VL_adapter/VL-T5/hf_models/uie-base-en",
        #     config=config,
        #     **kwargs
        #     )
        #     # model_name_or_path = "/home/liqingyuan/VL_adapter/VL-T5/hf_models/uie-base-en"
        #     # model = AutoModelForSeq2SeqLM.from_pretrained(
        #     #     model_name_or_path,
        #     #     from_tf=bool(".ckpt" in model_name_or_path),
        #     #     config=config,
        #     #     cache_dir=None,
        #     #     revision=None,
        #     #     use_auth_token=None,
        #     #     mirror='tuna',
        #     #  )
        #     # return model
        # else:
        model = model_class.from_pretrained(
            model_name,
            config=config,
            **kwargs )

        return model

    def print_trainable_params_percentage(self, model):
        # if "bart-base" in self.args.backbone:
        #     orig_param_size = 139420416
        # elif "t5-base" in self.args.backbone:
        #     orig_param_size = 222903552
        # else:
        #     print(f"Don't know the parameters number of this {self.args.backbone}")
        #     orig_param_size = -1

        orig_param_size = sum(p.numel() for p in model.parameters())

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        # for n,p in model.named_parameters(): 
        #     if not p.requires_grad:
        #         print(n)
        # exit()
        trainable_size = count_parameters(model)

        percentage = trainable_size / orig_param_size * 100

        print(f"Trainable param percentage: {percentage:.2f}% ({trainable_size}/{orig_param_size})")

        return percentage

    def freeze_whole_model(self):
        for n, p in self.model.named_parameters():
            p.requires_grad = False

    def partial_eval(self):
        # the purpose is to fix some of the norm statistics
        model = self.model.module if self.args.distributed else self.model

        def LM_LN_eval(model):
            for name, sub_module in model.named_modules():
                if "adapter" in name: # skip all adapters modules
                    continue
                if isinstance(sub_module, (modeling_t5.T5Stack, modeling_t5.JointEncoder)):
                    # print(f"Change {name} to eval mode...")
                    sub_module.eval()

        def only_LN_eval(model):
            for name, sub_module in model.named_modules():
                if "adapter" in name: # skip all adapters modules
                    continue
                if "visual_embedding" in name: # skip trainable parameters
                    continue
                if isinstance(sub_module, (T5LayerNorm, nn.LayerNorm)):
                    # print(f"Change {name} to eval mode...")
                    sub_module.eval() # freeze the LN statistics and dropout

        def only_BN_eval(model):
            for name, sub_module in model.named_modules():
                if "adapter" in name: # skip all adapters modules
                    continue
                if isinstance(sub_module, (nn.BatchNorm2d)):
                    # print(f"Change {name} to eval mode...")
                    sub_module.eval() # freeze the LN statistics and dropout

        if self.args.freeze_ln_statistics:
            only_LN_eval(model)

        if self.args.freeze_bn_statistics:
            only_BN_eval(model)

    def unfreeze_parameters(self):  
        # import pdb
        # pdb.set_trace()     
        targets = ["visual_embedding"]
        # unfreeze the parameters in targets anyway
        for n, p in self.model.named_parameters():
            if any(t in n for t in targets):
                p.requires_grad = True
                print(f"{n} is trainable...")
            # else:
            #     p.requires_grad = False

        if self.args.unfreeze_language_model:
            targets = ["lm_head", "shared"]
            for n, p in self.model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    # print(f"{n} is trainable...")
            for name, sub_module in self.model.named_modules():
                # if isinstance(sub_module, (modeling_bart.JointEncoder, modeling_bart.BartDecoder, modeling_t5.T5Stack, modeling_t5.JointEncoder,self.model.encoder,self.model.decoder)):
               
                # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                for param_name, param in sub_module.named_parameters():
                    if "vis_encoder" in param_name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                        # print(f"{name} is trainable...")


        if self.args.unfreeze_lm_head:
            targets = ["lm_head", "shared"] # shared and lm_head share the same weight
            for n, p in self.model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...")

        if self.args.use_lora:
            targets = ["lora", "bias"]
            for n, p in self.model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...")

        for name, sub_module in self.model.named_modules():
            if self.args.decoder_prompt_len > 0 or self.args.encoder_prompt_len > 0:
                if isinstance(sub_module, (PromptController)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.unfreeze_vis_encoder:
                if isinstance(sub_module, (CLIPResNetEncoder)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.unfreeze_vis_last_layer:
                if "visual.layer4" in name and "visual.layer4." not in name:
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_vis_adapter:
                if isinstance(sub_module, (VisualAdapter)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.unfreeze_layer_norms:
                if isinstance(sub_module, (T5LayerNorm, nn.LayerNorm)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.unfreeze_batch_norms:
                if isinstance(sub_module, (nn.BatchNorm2d)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_adapter or self.args.use_compacter or self.args.use_lradapter:
                if isinstance(sub_module, (AdapterController)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

                if isinstance(sub_module, (GateBlock,VStack)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_lm_head_adapter:
                if isinstance(sub_module, (OutputParallelAdapterLayer)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_hyperformer:
                if isinstance(sub_module, (TaskEmbeddingController, AdapterLayersHyperNetController, AdapterLayersOneHyperNetController)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if isinstance(sub_module, (GateBlock,VStack)):
                print(f"{name} is trainable...")
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True

    def create_tokenizer(self, **kwargs):
        from transformers import T5Tokenizer, BartTokenizer, T5TokenizerFast, BartTokenizerFast
        from tokenization import VLT5Tokenizer, VLT5TokenizerFast
        if 't5' in self.args.tokenizer:
            if self.args.use_vision:
                tokenizer_class = VLT5Tokenizer
                # tokenizer_class = T5TokenizerFast
            else:
                # tokenizer_class = T5Tokenizer
                tokenizer_class = T5TokenizerFast
        elif 'umie' in self.args.tokenizer:
            if self.args.use_vision:
                tokenizer_class = VLT5Tokenizer
                # tokenizer_class = T5TokenizerFast
            else:
                # tokenizer_class = T5Tokenizer
                tokenizer_class = T5TokenizerFast
        elif 'uie' in self.args.tokenizer:
                tokenizer = AutoTokenizer.from_pretrained(
                        self.args.backbone,
                        cache_dir=None,
                        use_fast=True,
                        revision=None,
                        use_auth_token=None,
                        max_length=self.args.max_text_length
                )
                return tokenizer
        elif 'bart' in self.args.tokenizer:
            tokenizer_class = BartTokenizer
            # tokenizer_class = BartTokenizerFast

        tokenizer_name = self.args.backbone
        
        # model_name_or_path = "/home/liqingyuan/VL_adapter/VL-T5/hf_models/uie-base-en"

        tokenizer = tokenizer_class.from_pretrained(
            self.args.backbone,
            # device_map="auto",            
            # do_lower_case=self.args.do_lower_case,
            **kwargs
            )

        return tokenizer

    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')

        lr_scheduler = None

        from transformers.optimization import AdamW, get_linear_schedule_with_warmup

        no_decay = ["bias", "LayerNorm.weight"]

        if 'adamw' in self.args.optim:
            # print([n for n, p in self.model.named_parameters() if ( (not any(nd in n for nd in no_decay)) and ("vis_encoder" in n) )])
            # import pdb
            # pdb.set_trace()
            if self.args.use_separate_optimizer_for_visual:
                
                # transformer's parameters
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if ( (not any(nd in n for nd in no_decay)) and ("vis_encoder" not in n) ) ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.lr,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if ( (any(nd in n for nd in no_decay)) and ("vis_encoder" not in n ))],
                        "weight_decay": 0.0,
                        "lr": self.args.lr,
                    },
                ]
                
                visn_model = self.model.encoder.vis_encoder
                if self.args.use_adam_for_visual:

                    vis_optimizer_grouped_parameters = [
                        {
                            "params": [p for n, p in visn_model.named_parameters() if not any(nd in n for nd in no_decay)],
                            "weight_decay": self.args.vis_weight_decay,
                            "lr": self.args.vis_lr,
                        },
                        {
                            "params": [p for n, p in visn_model.named_parameters() if any(nd in n for nd in no_decay)],
                            "weight_decay": 0.0,
                            "lr": self.args.vis_lr,
                        },
                    ]
                    optim = AdamW(
                        optimizer_grouped_parameters + vis_optimizer_grouped_parameters,
                        lr=self.args.lr,
                        # betas=(0.9, 0.98),
                        eps=self.args.adam_eps
                    )
                else:
                    optim = AdamW(
                        optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_eps
                    )
                    vis_optim = torch.optim.SGD(
                        visn_model.parameters(), 
                        self.args.vis_lr,
                        momentum=0,
                        weight_decay=self.args.vis_weight_decay
                    )

                    optim = FusedOptimizer([optim, vis_optim])

            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]
                optim = AdamW(optimizer_grouped_parameters,
                            lr=self.args.lr, eps=self.args.adam_eps)

        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            # if self.include_vis_encoder:
            #     trainable_parameters = trainable_parameters + list(self.vis_encoder.parameters())

            optim = self.args.optimizer(optimizer_grouped_parameters, self.args.lr)
        
        batch_per_epoch = len(self.train_loader)
        t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
        warmup_ratio = self.args.warmup_ratio
        warmup_iters = int(t_total * warmup_ratio)
        if self.verbose:
            print("Batch per epoch: %d" % batch_per_epoch)
            print("Total Iters: %d" % t_total)
            print('Warmup ratio:', warmup_ratio)
            print("Warm up Iters: %d" % warmup_iters)

        lr_scheduler = get_linear_schedule_with_warmup(optim, warmup_iters, t_total)

        return optim, lr_scheduler

    def load_checkpoint(self, ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')

        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            pprint(results)

    def init_weights(self):

        def init_bert_weights(module):
            """ Initialize the weights."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=1)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.model.apply(init_bert_weights)
        self.model.init_weights()

    def predict(self):
        pass

    def evaluate(self):
        pass

    def save(self, name):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)

        if use_deepspeed:
            self.model.save_checkpoint(self.args.output+'/'+name)
        else:
            torch.save(self.model.state_dict(), os.path.join(self.args.output, "%s.pth" % name))

    def load(self, path, loc=None):
        if loc is None and hasattr(self.args, 'gpu'):
            loc = f'cuda:{self.args.gpu}'
        state_dict = torch.load("%s.pth" % path, map_location=loc)

        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', path)
            pprint(results)
