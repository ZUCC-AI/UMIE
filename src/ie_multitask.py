
from trainer_base import TrainerBase
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version
from extraction.record_schema import RecordSchema
from seq2seq.constraint_decoder import get_constraint_decoder
from extraction.noiser.spot_asoc_noiser import SpotAsocNoiser
from ie_multitask_model import UMIEMultiTask
from extraction.extraction_metrics import get_extract_metrics
from seq2seq.data_collator import (
    DataCollatorForMetaSeq2Seq,
    DynamicSSIGenerator,
)
from extraction import constants
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint, pformat
from copy import deepcopy
from functools import partial
from param import parse_args
from functools import partial

import deepspeed
import ie_multitask_data

from utils import LossMeter
import wandb


proj_dir = Path(__file__).resolve().parent.parent


_use_amp = False
from torch.cuda.amp import autocast
scaler = torch.cuda.amp.GradScaler(enabled=_use_amp)

import warnings

warnings.filterwarnings('ignore')

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex


class Record:
    def __init__(self,record,record_type):
        self.record = record
        self.record_type = record_type
    
class AllRecordSchema:

    mner_record_schema = RecordSchema.read_from_file(f"/data2/datasets/MNER/img_mnersnap/record.schema")

    mnre_v2_record_schema = RecordSchema.read_from_file(f"/data2/datasets/MNRE/mnre_v2/record.schema")
    
    mnre_v1_record_schema = RecordSchema.read_from_file(f"/data2/datasets/MNRE/mnre_v1/record.schema")

    ace05_record_schema = RecordSchema.read_from_file("/data2/datasets/ACE05/record.schema")
    

    m2e2_record_schema =  RecordSchema.read_from_file("/data2/datasets/m2e2/record.schema")

    
    swig_record_schema = RecordSchema.read_from_file("/data2/datasets/swig/record.schema")

    allschema ={
        "mner_mner2015": mner_record_schema,
        "mner_mnersnap": mner_record_schema,
        "mner_mner2017": mner_record_schema,
        "relation_mnre_v2":mnre_v2_record_schema,
        "relation_mnre_v1":mnre_v1_record_schema,
        "event_ace05_all":ace05_record_schema,
        "event_ace05_trigger":ace05_record_schema,
        "event_ace05_arg":ace05_record_schema,
        "event_m2e2_all": m2e2_record_schema,
        "event_m2e2_trigger":m2e2_record_schema,
        "event_m2e2_arg":m2e2_record_schema,
        "event_swig_all":ace05_record_schema,
        "event_swig_trigger":ace05_record_schema,
        "event_swig_arg": ace05_record_schema,
    }
    
    # ace05_record_schema = mner_record_schema
    

    
    

    
def get_loader(args, tokenizer, data_collector):
    feat_dim_dict = {
        "RN50": 2048,
        "RN101": 2048,
        "RN50x4": 2560,
        "ViT": 768
    }
    args.feat_dim = feat_dim_dict[args.feature_type] #2048
    import mner_clip_data as mner_data
    import mnre_clip_data as mnre_data
    import swig_event_clip_data as swig_event
    import swig_clip_data as swig_data
    import m2e2_clip_data as m2e2_data
    import ace05_clip_data as ace05_data
    import voa_clip_data as voa_data
    
    alldataset = {
        "mner_mner2015": mner_data,
        "mner_mnersnap": mner_data,
        "mner_mner2017": mner_data,
        "relation_mnre_v2": mnre_data,
        "relation_mnre_v1":mnre_data,
        "event_ace05_all":ace05_data,
        "event_ace05_trigger":ace05_data,
        "event_ace05_arg": ace05_data,
        "event_m2e2_all": m2e2_data,
        "event_m2e2_trigger":m2e2_data,
        "event_m2e2_arg":m2e2_data,
        "event_swig_all":swig_event,
        "event_swig_trigger":swig_event,
        "event_swig_arg": swig_event,
        # "swig": swig_data,
    }
    args_dict = {}
    for key, value in alldataset.items():
        if key.startswith("mner"):
            mner_args = deepcopy(args)
            mner_args.prompt = "Please extract the following entity type:"
            # mner_args.prompt = ""
            args_dict[key] = mner_args
        elif key.startswith("relation"):
            mnre_args = deepcopy(args)
            mnre_args.prompt = "Please extract the following relation between:"
            # mnre_args.prompt = ""
            args_dict[key] = mnre_args
        elif key.endswith("-trigger"):
            event_trigger_args = deepcopy(args)
            event_trigger_args.prompt = "Extract event trigger: "
            # event_trigger_args.prompt = ""
            args_dict[key] = event_trigger_args
        elif key.endswith("-arg"):
            event_argument_args = deepcopy(args)
            event_argument_args.prompt = "Extract event argument: "
            # event_argument_args.prompt = ""
            args_dict[key] = event_argument_args
        else:
            event_args = deepcopy(args)
            event_args.prompt = "Extract event trigger and argument: "
            # event_args.prompt = ""
            args_dict[key] = event_args

      
    
    task_num = 0
    #args.use_tasks_prompts: False
    val_num_workers = 4
    train_loaders = []
    train_datasets = [] 

    val_loader = {}
    test_loader = {}
    if args.epochs > 0:
        for key, value in alldataset.items():
            if key in args.tasks:           
                print(f'Building {key} train loader at GPU {args.gpu}')
                train_loader, train_dataset = alldataset[key].get_loader(
                    args_dict[key],
                    mode='train',
                    batch_size=args.batch_size,
                    distributed=args.distributed, 
                    gpu=args.gpu,
                    workers=args.num_workers,
                    topk=args.train_topk,
                    collate_fn = data_collector,
                    tokenizer = tokenizer,
                    dataset_prefix =key.split("_")[1] if key.endswith("-trigger") or key.endswith("-arg") else key.split("_",1)[1],
                    task = key
                ) 
                task_num += 1
                train_loaders.append(train_loader)
                # train_datasets.append(train_dataset)
            if args.gpu == 0:
                if key in args.eval_tasks:
                    print(f'Building {key} val loader at GPU {args.gpu}')
                    print("key")
                    data_val_loader, val_dataset = alldataset[key].get_loader(
                        args_dict[key],
                        mode='val', 
                        batch_size=32,
                        distributed= False,
                        gpu=args_dict[key].gpu,
                        workers=val_num_workers,
                        topk=args.valid_topk,
                        collate_fn = data_collector,
                        tokenizer = tokenizer,
                        dataset_prefix =key.split("_")[1] if key.endswith("-trigger") or key.endswith("-arg") else key.split("_",1)[1],
                        task = key
                    )
                    val_loader[key] = data_val_loader
                    
                if key in args.eval_tasks:
                    print(f'Building {key} test loader at GPU {args.gpu}')
                    data_test_loader, test_dataset = alldataset[key].get_loader(
                        args_dict[key],
                        mode='test', 
                        batch_size=32,
                        distributed=False, 
                        gpu=args.gpu,
                        workers=val_num_workers,
                        topk=args.valid_topk,
                        collate_fn =data_collector,
                        tokenizer = tokenizer,
                        dataset_prefix =key.split("_")[1] if key.endswith("-trigger") or key.endswith("-arg") else key.split("_",1)[1],
                        task = key
                    )
                    test_loader[key] = data_test_loader
        

    if len(train_datasets) == 0:   
        train_loader = ie_multitask_data.MultitaskLoader(
            train_loaders,
            sampling=args.multitask_sampling,
            verbose=args.gpu==0
        )
    else:
        from torch.utils.data import DataLoader, Dataset, Sampler
        from torch.utils.data.distributed import DistributedSampler

        concat_ds = torch.utils.data.ConcatDataset(train_datasets)
        train_sampler = DistributedSampler(concat_ds)
        train_loader = DataLoader(
            concat_ds, batch_size=args.batch_size, shuffle=(not args.distributed),
            num_workers=args.num_workers, pin_memory=False, sampler=train_sampler,
            collate_fn = data_collector)
    
    
    return train_loader, val_loader, test_loader, task_num

class Trainer(TrainerBase):
    def __init__(self,
                args,
                train=True):
        
        super().__init__(args)
        

        model_kwargs = {}
        model_class = UMIEMultiTask

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()

        if 't5' in self.args.tokenizer or 'umie' in self.args.tokenizer : 
            to_add_special_token = list()
            for special_token in [constants.type_start, constants.type_end, constants.span_start, constants.spot_prompt, constants.asoc_prompt]:
                if special_token not in self.tokenizer.get_vocab():
                    to_add_special_token += [special_token]
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": to_add_special_token})
            self.model = self.create_model(model_class, config, **model_kwargs)
            self.model.resize_token_embeddings(len(self.tokenizer))

        negative_sampler = partial(DynamicSSIGenerator,tokenizer=self.tokenizer, positive_rate=1,  negative=-1, ordered_prompt=False)
        
        negative_sampler_dict = {}
        for key, schema in AllRecordSchema.allschema.items():
            negative_sampler_dict[key] = negative_sampler(schema = schema)
        self.spot_asoc_nosier = SpotAsocNoiser(
                    spot_noise_ratio=0.1,
                    asoc_noise_ratio=0.1,
                    null_span='<extra_id_6>',
        )

        data_collator = DataCollatorForMetaSeq2Seq(
            tokenizer = self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            max_length=256,
            max_prefix_length=-1,
            max_target_length=self.args.gen_max_length,
            spot_asoc_nosier=self.spot_asoc_nosier,
            decoding_format='spotasoc',
            negative_sampler = negative_sampler_dict
        )
        
    
        # data_collator_dict = {}
        # for key, negative_sampler in negative_sampler_dict.items():
        #     data_collator_dict[key] = data_collator(negative_sampler=negative_sampler)
          
        self.train_loader, self.val_loader, self.test_loader, self.task_num = get_loader(args, self.tokenizer, data_collator)
 

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)
        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        if not self.args.use_deepspeed:
            self.model = self.model.to(f"cuda:{args.gpu}")

        self.freeze_whole_model() # freeze whole parameters first
        self.unfreeze_parameters() # unfreeze selected parameters

   
        if args.use_lora:
            from peft import LoraConfig, get_peft_model,prepare_model_for_int8_training, TaskType

            lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
            )
            # prepare int-8 model for training
            if "xl" in args.backbone:
                self.model = prepare_model_for_int8_training(self.model)

            # add LoRA adaptor
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        # Optimizer

        # if args.use_single_prompt:
        #     from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType

        #     peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)

        #     self.model = get_peft_model(self.model, peft_config)
        #     self.model.print_trainable_parameters()

        # self.freeze_whole_model() # freeze whole parameters first
        # self.unfreeze_parameters() # unfreeze selected parameters


        # print(self.model)
        self.percent_updated_parameters = self.print_trainable_params_percentage(self.model)

        if self.args.use_deepspeed:
            batch_per_epoch = len(self.train_loader)
            t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            warmup_ratio = self.args.warmup_ratio
            warmup_iters = int(t_total * warmup_ratio)

            ds_config={
                "train_batch_size": self.args.batch_size * self.args.world_size * self.args.gradient_accumulation_steps,
                "steps_per_print": 1000,
                "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.lr,
                    "betas": [
                    0.9,
                    0.999
                    ],
                    "eps": 5e-8,
                    "weight_decay": 0.1
                }
                },
                "tensorboard": {
                    "enabled": True,
                    "output_path": "output/ds_logs/",
                    "job_name": "training_muie"
                },
                "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr":  args.lr,
                    "warmup_num_steps": warmup_iters,
                    # "total_num_steps": t_total
                }
                },
                "gradient_clipping": self.args.clip_grad_norm,
                "bfloat16":{
                    "enabled": self.args.bfp16,
                },
                 "zero_optimization": {
                     "stage": 1
                 }
                
               # "zero_optimization": {
              #      "stage": 3,
               #     "offload_optimizer": {
               #         "device": "cpu",
               #         "pin_memory": True
               #     },
               #     "allgather_partitions": True,
               #     "allgather_bucket_size": 2e8,
               #     "reduce_scatter": True,
               #     "reduce_bucket_size": 2e8,
               #     "overlap_comm": True,
               #     "contiguous_gradients": True
               # }
            }

            print(ds_config)
            self.model, self.optim, _, self.lr_scheduler =deepspeed.initialize(args=args,
                            model=self.model,
                            model_parameters=model_parameters,
                            config = ds_config)


        else:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                    find_unused_parameters=True
                                    )
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()
       
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

        self.to_remove_token_list = list()
        if self.tokenizer.bos_token:
            self.to_remove_token_list += [self.tokenizer.bos_token]
        if self.tokenizer.eos_token:
            self.to_remove_token_list += [self.tokenizer.eos_token]
        if self.tokenizer.pad_token:
            self.to_remove_token_list += [self.tokenizer.pad_token]
    

    def train(self):
        if self.verbose:
            mner2015_loss_meter = LossMeter()
            mnersnap_loss_meter = LossMeter()
            mner2017_loss_meter = LossMeter()

            mnre_v1_loss_meter = LossMeter()
            mnre_v2_loss_meter = LossMeter()
            swig_event_loss_meter = LossMeter()
            ace05_evt_loss_meter = LossMeter()
            swig_loss_meter = LossMeter()
            total_loss_meter = LossMeter()

            best_mner2015_valid = 0.
            best_mner2015_epoch = 0

            # gqa
            best_mnersnap_valid = 0
            best_mnersnap_epoch = 0
            
            best_mner2017_valid = 0
            best_mner2017_epoch = 0

            best_mnre_v2_valid = 0
            best_mnre_v2_epoch = 0
            
            best_mnre_v1_valid = 0
            best_mnre_v1_epoch = 0

            best_m2e2_valid = 0
            best_m2e2_epoch = 0   
        
            best_m2e2_trigger_valid = 0
            best_m2e2_trigger_epoch = 0 
            

            cur_vaild = 0
            
            best_vaild = 0 
            best_epoch = 0
            # assert 't5'in self.args.backbone
            # self.setup_wandb()
            self.wandb_initialized = False
            if not self.wandb_initialized:

                if 't5' in self.args.backbone:
                    project_name = "VLT5_multitask"
                elif 'bart' in self.args.backbone:
                    project_name = "VLBart_multitask"
                elif 'umie' in self.args.backbone:
                    project_name = "VLUIE_multitask"

                wandb.init(project=project_name)
                wandb.run.name = self.args.run_name
                wandb.config.update(self.args)
                wandb.watch(self.model)
                wandb.log(
                    {"percent of updated parameters (%)": self.percent_updated_parameters}
                )

                src_dir = Path(__file__).resolve().parent
                base_path = str(src_dir.parent)
                src_dir = str(src_dir)
                wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

                self.wandb_initialized = True


        if self.args.distributed:
            dist.barrier()

        global_step = 0
        for epoch in range(self.args.epochs):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            # self.partial_eval()

            self.train_loader.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=250)

            epoch_results = {
                'loss': 0.,

            }

            task_counter = {
                "mner_mner2015": 0,
                "mner_mnersnap": 0,
                "mner_mner2017": 0,
                "relation_mnre_v2": 0,
                "relation_mnre_v1": 0,
                "event_ace05_all":0,
                "event_ace05_trigger":0,
                "event_ace05_arg": 0,
                "event_m2e2_all": 0,
                "event_m2e2_trigger":0,
                "event_m2e2_arg":0,
                "event_swig_all":0,
                "event_swig_trigger":0,
                "event_swig_arg": 0,
                "total":0
                # "swig": swig_data,
            }



            for step_i, batch in enumerate(self.train_loader):

                task = batch['task']
                # task = 'total'
                task_counter[task] += 1

                batch['log_train_accuracy'] = self.args.log_train_accuracy

                if _use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=_use_amp):
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                    
                    loss = results['loss']
   
                    scaler.scale(loss).backward()
                    scaler.step(self.optim)
                    scaler.update()
                    self.lr_scheduler.step()

                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                    loss = results['loss']

                    if self.args.use_deepspeed:
                        self.model.backward(loss)
                    else:
                        # print("loss.backward()")
                        loss.backward()
                        
                    loss = loss.detach()

                    
                    if self.args.use_deepspeed:
                        self.model.step()
                    else:
                        # print(self.optim)
                        self.optim.step()
                        # print
                        self.lr_scheduler.step()

     
                global_step += 1

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

                lr = self.lr_scheduler.get_lr()[0]
                
                if self.verbose:
                    if task == 'mner_mner2015':
                        mner2015_loss_meter.update(loss.item())
                    if task == 'mner_mner2017':
                        mner2017_loss_meter.update(loss.item())
                    elif task == 'mner_mnersnap':
                        mnersnap_loss_meter.update(loss.item())
                    elif task == 'relation_mnre_v2':
                        mnre_v2_loss_meter.update(loss.item())
                    elif task == 'relation_mnre_v1':
                        mnre_v1_loss_meter.update(loss.item())
                    elif task == 'event_swig_all':
                        swig_event_loss_meter.update(loss.item())
                    elif task == 'event_ace05_all':
                        ace05_evt_loss_meter.update(loss.item())
                    elif task == 'total':
                        total_loss_meter.update(loss.item())

                    desc_str = f'Epoch {epoch} | LR {lr:.6f}'

                    desc_str += f" |"
                    if 'mner_mner2015' in self.args.tasks:
                        desc_str += f" mner2015 {task_counter['mner_mner2015']}"
                    if 'mner_mner2017' in self.args.tasks:
                        desc_str += f" mner2017 {task_counter['mner_mner2017']}"
                    if 'mner_mnersnap' in self.args.tasks:
                        desc_str += f" mnersnap {task_counter['mner_mnersnap']}"
                    if 'relation_mnre_v1' in self.args.tasks:
                        desc_str += f" relation_mnre_v1 {task_counter['relation_mnre_v1']}"
                    if 'relation_mnre_v2' in self.args.tasks:
                        desc_str += f" relation_mnre_v2 {task_counter['relation_mnre_v2']}"
                    if 'event_swig_all' in self.args.tasks:
                        desc_str += f" event_swig {task_counter['event_swig_all']}"
                    if 'event_ace05_all' in self.args.tasks:
                        desc_str += f" event_ace05 {task_counter['event_ace05_all']}"
                    if len(mner2015_loss_meter) > 0:
                        desc_str += f' | mner2015 Loss {mner2015_loss_meter.val:4f}'
                    if len(mner2017_loss_meter) > 0:
                        desc_str += f' | mner2017 Loss {mner2017_loss_meter.val:4f}'
                    if len(mnersnap_loss_meter) > 0:
                        desc_str += f' | mnersnap Loss {mnersnap_loss_meter.val:.3f}'
                    if len(mnre_v1_loss_meter) > 0:
                        desc_str += f' | mnre_v1 Loss {mnre_v1_loss_meter.val:.3f}'    
                    if len(mnre_v2_loss_meter) > 0:
                        desc_str += f' | mnre_v2 Loss {mnre_v2_loss_meter.val:.3f}'
                    if len(swig_event_loss_meter) > 0:
                        desc_str += f' | swig_evt Loss {swig_event_loss_meter.val:.3f}'
                    if len(ace05_evt_loss_meter) > 0:
                        desc_str += f' | ace-evt Loss {ace05_evt_loss_meter.val:.3f}'
                    if len(swig_loss_meter) > 0:
                        desc_str += f' | swig Loss {swig_loss_meter.val:.3f}'
                    if len(total_loss_meter) > 0:
                        desc_str += f' | total Loss {total_loss_meter.val:.3f}'


                    pbar.set_description(desc_str)
                    pbar.update(1)

                if self.args.distributed:
                    dist.barrier()

            if self.verbose:
                pbar.close()
                
            print()
            if self.args.gpu==0 and self.verbose and epoch >= 1:
                # print("Epoch%02d" % (epoch + 1))
                # self.save("Epoch%02d" % (epoch + 1))

                # self.save("Epoch%02d" % (epoch + 1))
                log_str = ''
                wandb_log_dict = {}

                if 'mner_mner2015' in self.args.eval_tasks:
                    # MNER
                    mner2015_test_loader = self.test_loader['mner_mner2015']
                    mner2015_test_results, mner2015_test_predictions= self.evaluate(mner2015_test_loader,AllRecordSchema.mner_record_schema,"entity")
                    print("--------------------------------------------------------")
                    print(loss.item())
                    print("--------------------------------------------------------")
                    overall_F1 = mner2015_test_results['overall-F1']
                    cur_vaild += overall_F1
                    if overall_F1 > best_mner2015_valid or epoch == 0:
                        best_mner2015_valid = overall_F1
                        best_mner2015_epoch = epoch

                    log_str += f"mner2015"
                    log_str += "\nEpoch %d: Valid overall-F1 %0.2f " % (epoch, overall_F1)
                    log_str += "\nEpoch %d: Best overall-F1 %0.2f\n" % (best_mner2015_epoch, best_mner2015_valid)
                    wandb_log_dict['mner2015/Valid/overall-F1'] = best_mner2015_valid

                if 'mner_mnersnap' in self.args.eval_tasks:
                    # GQA
                    mnersnap_test_loader = self.test_loader['mner_mnersnap']
                    mnersnap_test_results, mnersnap_test_predictions= self.evaluate(mnersnap_test_loader,AllRecordSchema.mner_record_schema,"entity")
                    print("--------------------------------------------------------")
                    print(mnersnap_test_results)
                    print("--------------------------------------------------------")

                    overall_F1 = mnersnap_test_results['overall-F1']
                    cur_vaild += overall_F1

                    if overall_F1 > best_mnersnap_valid or epoch == 0:
                        best_mnersnap_valid = overall_F1
                        best_mnersnap_epoch = epoch
                    log_str += f"mnersnap"
                    log_str += "\nEpoch %d: Valid overall-F1 %0.2f " % (epoch, overall_F1)
                    log_str += "\nEpoch %d: Best overall-F1 %0.2f\n" % (best_mnersnap_epoch, best_mnersnap_valid)
                    wandb_log_dict['mnersnap/Valid/overall-F1'] = best_mnersnap_valid
                    
                if 'mner_mner2017' in self.args.eval_tasks:
                    mner2017_test_loader = self.test_loader['mner_mner2017']
                    mner2017_test_results, mner2017_test_predictions= self.evaluate(mner2017_test_loader, AllRecordSchema.mner_record_schema,"entity")
                    # print("--------------------------------------------------------")
                    # print(mnersnap_val_results)
                    print("--------------------------------------------------------")
                    print(mner2017_test_results)
                    print("--------------------------------------------------------")

                    overall_F1 = mner2017_test_results['overall-F1']
                    cur_vaild += overall_F1

                    if overall_F1 > best_mner2017_valid or epoch == 0:
                        best_mner2017_valid = overall_F1
                        best_mner2017_epoch = epoch
                        # self.save("VQA_BEST")
                    log_str += f"mner2017"
                    log_str += "\nEpoch %d: Valid overall-F1 %0.2f " % (epoch, overall_F1)
                    log_str += "\nEpoch %d: Best overall-F1 %0.2f\n" % (best_mner2017_epoch, best_mner2017_valid)
                    wandb_log_dict['mnersnap/Valid/overall-F1'] = best_mner2017_valid
               
                if 'relation_mnre_v1' in self.args.eval_tasks:
                    # GQA
                    mnre_v1_test_loader = self.test_loader['relation_mnre_v1']
                    mnre_v1_test_results, mnre_v1_test_predictions= self.evaluate(mnre_v1_test_loader,AllRecordSchema.mnre_v1_record_schema,"relation" )
                    # print("--------------------------------------------------------")
                    # print(mnre_v1_val_results)
                    print("--------------------------------------------------------")
                    print(mnre_v1_test_results)
                    print("--------------------------------------------------------")

                    overall_F1 = mnre_v1_test_results['overall-F1']
                    cur_vaild += overall_F1
                    
                    if overall_F1 > best_mnre_v1_valid or epoch == 0:
                        best_mnre_v1_valid = overall_F1
                        best_mnre_v1_epoch = epoch
                        # self.save("VQA_BEST")
                    log_str += f"mner_v1"
                    log_str += "\nEpoch %d: Valid overall-F1 %0.2f " % (epoch, overall_F1)
                    log_str += "\nEpoch %d: Best overall-F1 %0.2f\n" % (best_mnre_v1_epoch, best_mnre_v1_valid)
                    wandb_log_dict['mnre_v1/Valid/overall-F1'] = best_mnre_v1_valid
                    
                if 'relation_mnre_v2' in self.args.eval_tasks:
                    mnre_v2_test_loader = self.test_loader['relation_mnre_v2']
                    mnre_v2_test_results, mnre_v2_test_predictions= self.evaluate(mnre_v2_test_loader, AllRecordSchema.mnre_v2_record_schema, "relation")
                    # print("--------------------------------------------------------")
                    # print(mnre_v2_val_results)
                    print("--------------------------------------------------------")
                    print(mnre_v2_test_results)
                    print("--------------------------------------------------------")

                    overall_F1 = mnre_v2_test_results['overall-F1']
                    cur_vaild += overall_F1
                    
                    if overall_F1 > best_mnre_v2_valid or epoch == 0:
                        best_mnre_v2_valid = overall_F1
                        best_mnre_v2_epoch = epoch
                        # self.save("VQA_BEST")
                    log_str += f"mner_v2"
                    log_str += "\nEpoch %d: Valid overall-F1 %0.2f " % (epoch, overall_F1)
                    log_str += "\nEpoch %d: Best overall-F1 %0.2f\n" % (best_mnre_v2_epoch, best_mnre_v2_valid)
                    wandb_log_dict['mnre_v2/Valid/overall-F1'] = best_mnre_v2_valid
                  

                if 'event_m2e2_all' in self.args.eval_tasks:
                    # GQA
                    m2e2_test_loader = self.test_loader['event_m2e2_all']
                    m2e2_test_results, m2e2_test_predictions= self.evaluate(m2e2_test_loader,AllRecordSchema.m2e2_record_schema,"event")
                    print("--------------------------------------------------------")
                    print(m2e2_test_results)
                    print("--------------------------------------------------------")

                    overall_F1 = m2e2_test_results['overall-F1']
                    cur_vaild += overall_F1

                    if overall_F1 > best_m2e2_valid or epoch == 0:
                        best_m2e2_valid = overall_F1
                        best_m2e2_epoch = epoch
                        # self.save("VQA_BEST")
                    log_str += f"m2e2"
                    log_str += "\nEpoch %d: Valid overall-F1 %0.2f " % (epoch, overall_F1)
                    log_str += "\nEpoch %d: Best overall-F1 %0.2f\n" % (best_m2e2_epoch, best_m2e2_valid)
                    wandb_log_dict['m2e2/Valid/overall-F1'] = best_m2e2_valid
                    log_str += "\nEpoch %d:  spot-F1 %0.2f " % (epoch, m2e2_test_results["spot-F1"])
                    log_str += "\nEpoch %d:  asoc-F1 %0.2f\n" % (epoch, m2e2_test_results["asoc-F1"])
                    
                if 'event_m2e2_trigger' in self.args.eval_tasks:
    
                    m2e2_trigger_test_loader = self.val_loader['event_m2e2_trigger']
                    m2e2_trigger_test_results, well_formed_list_trigger = self.evaluate(m2e2_trigger_test_loader, AllRecordSchema.m2e2_record_schema, "event_trigger")
                    print("---------------------------m2e2_trigger_test_loader-----------------------------")
                    print(m2e2_trigger_test_results)
                    print("---------------------------m2e2_trigger_test_loader-----------------------------")
                    
                    overall_F1 = m2e2_trigger_test_results['spot-F1']
                    cur_vaild += overall_F1

                    if overall_F1 > best_m2e2_trigger_valid or epoch == 0:
                        best_m2e2_trigger_valid = overall_F1
                        best_m2e2_trigger_epoch = epoch
                        # self.save("VQA_BEST")
                    log_str += f"m2e2"
                    log_str += "\nEpoch %d: Valid overall-F1 %0.2f " % (epoch, overall_F1)
                    log_str += "\nEpoch %d: Best overall-F1 %0.2f\n" % (best_m2e2_trigger_epoch, best_m2e2_trigger_valid)
                    wandb_log_dict['m2e2/Valid/overall-F1'] = best_m2e2_trigger_epoch
        
                if 'event_m2e2_arg' in self.args.eval_tasks:
                    # GQA
                    m2e2_arg_test_loader = self.val_loader['event_m2e2_arg']
                    m2e2_arg_test_results, well_formed_list_arg = self.evaluate(m2e2_arg_test_loader, AllRecordSchema.m2e2_record_schema, "event_arg")
                    print("---------------------------event_m2e2_arg-----------------------------")
                    print(m2e2_arg_test_results['asoc-F1'])
                    print("---------------------------event_m2e2_arg-----------------------------")

                
                    overall_F1 = m2e2_arg_test_results['overall-F1']
                    cur_vaild += overall_F1

                    if overall_F1 > best_m2e2_valid or epoch == 0:
                        best_m2e2_arg_valid = overall_F1
                        best_m2e2_arg_epoch = epoch
                        # self.save("VQA_BEST")
                    log_str += f"m2e2"
                    log_str += "\nEpoch %d: Valid overall-F1 %0.2f " % (epoch, overall_F1)
                    log_str += "\nEpoch %d: Best overall-F1 %0.2f\n" % (best_m2e2_epoch, best_m2e2_valid)
                    wandb_log_dict['m2e2/Valid/overall-F1'] = best_m2e2_valid
                    
                if 'event_m2e2_two_stage' in self.args.eval_tasks:
                    from extraction.scorer import Metric, RecordMetric, OrderedRecordMetric

                    spot_metric = Metric()
                    asoc_metric = Metric()
                    for instance in well_formed_list_trigger:
                        spot_metric.count_instance(instance['gold_spot'], instance['pred_spot'])

                        
                    for instance_spot, instance_asoc in zip(well_formed_list_trigger, well_formed_list_arg):
                        print(instance_spot['pred_spot']== instance_spot['gold_spot'],instance_spot['pred_spot'],instance_spot['gold_spot'])
                        if len(instance_spot['pred_spot']) != 0 and len(instance_asoc['pred_asoc']) != 0:
                            for i in range(0,len(instance_asoc['pred_asoc'])):
                                print("inst origin asoc", instance_asoc['pred_asoc'][i])
                                tmp_tul = (instance_spot['pred_spot'][0][0], instance_asoc['pred_asoc'][i][1],instance_asoc['pred_asoc'][i][2])
                                instance_asoc['pred_asoc'][i] = tmp_tul
                                print("inst modify asoc", instance_asoc['pred_asoc'][i])

                        elif len(instance_spot['pred_spot']) == 0 and len(instance_asoc['pred_asoc'])!= 0:
                            for i in range(0,len(instance_asoc['pred_asoc'])):
                                tmp_tul = ("none", instance_asoc['pred_asoc'][i][1],instance_asoc['pred_asoc'][i][2])
                                print("none origin asoc", instance_asoc['pred_asoc'][i])
                                instance_asoc['pred_asoc'][i] = tmp_tul     
                                print("none modify asoc", instance_asoc['pred_asoc'][i], "gold asoc:", instance_asoc['gold_asoc'])
        
                           
                        print("pred_asoc",instance_asoc['pred_asoc'], "gold_asoc",instance_asoc['gold_asoc'])
                        asoc_metric.count_instance(instance_asoc['gold_asoc'], instance_asoc['pred_asoc'])
                    spot_result = spot_metric.compute_f1(prefix='spot-')
                    asoc_result = asoc_metric.compute_f1(prefix='asoc-')

                    overall_f1 = spot_result.get('spot-F1', 0.) + asoc_result.get('asoc-F1', 0.)
                    # print(counter)
                    result = {'overall-F1': overall_f1}
                    result.update(spot_result)
                    result.update(asoc_result)
                    print(result)
                                
                    
                if 'ace05-evt' in self.args.eval_tasks:
                    ace05_evt_test_loader = self.test_loader['ace05-evt']
                    ace05_evt_test_results, mace05_evt_test_predictions  = self.ace05_evt_evaluate(ace05_evt_test_loader)
                    print("--------------------------------------------------------")
                    print(ace05_evt_test_results)
                    print("--------------------------------------------------------")

                    overall_F1 = ace05_evt_test_results['overall-F1']
                    cur_vaild += overall_F1

                    if overall_F1 > best_m2e2_valid or epoch == 0:
                        best_ace05_evt_valid = overall_F1
                        best_ace05_evt_epoch = epoch
                        # self.save("VQA_BEST")
                    log_str += f"ace05-evt"
                    log_str += "\nEpoch %d: Valid overall-F1 %0.2f " % (epoch, overall_F1)
                    log_str += "\nEpoch %d: Best overall-F1 %0.2f\n" % (best_m2e2_epoch, best_m2e2_valid)
                    wandb_log_dict['ace05-evt/Valid/overall-F1'] = best_m2e2_valid
                    log_str += "\nEpoch %d:  spot-F1 %0.2f " % (epoch, ace05_evt_test_results["spot-F1"])
                    log_str += "\nEpoch %d:  asoc-F1 %0.2f\n" % (epoch, ace05_evt_test_results["asoc-F1"])
                
           
                
                if cur_vaild > best_vaild or epoch == 0:
                    # self.save("BEST")
                    best_vaild = cur_vaild
                    best_epoch = epoch
            
                log_str += f"all"
                log_str += "\nEpoch %d: Valid overall-F1 %0.2f " % (epoch, cur_vaild)
                log_str += "\nEpoch %d: Best overall-F1 %0.2f\n" % (best_epoch, best_vaild)
                wandb_log_dict['all/overall-F1'] = best_vaild
                wandb.log(wandb_log_dict, step=epoch)

                print(log_str)

            if self.args.distributed:
                dist.barrier()

   
        if self.args.distributed:
            dist.barrier()
            exit()

    def predict(self, loader, dump_path=None):
        """
        Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        with torch.no_grad():

            predictions = []
            targets = []

            gen_kwargs = {}
         
            gen_kwargs['num_beams'] = self.args.num_beams
            gen_kwargs['max_length'] = self.args.gen_max_length

            for i, batch in enumerate(tqdm(loader, ncols=120, desc="Prediction")):

                if self.args.distributed:
                    results = self.model.module.test_step(
                        batch,
                        # self.constraint_decoder,
                        **gen_kwargs)
                else:
                    results = self.model.test_step(
                        batch,
                        # self.constraint_decoder,
                        **gen_kwargs)

                predictions.extend(results['pred'])

                if 'labels' in batch:
                    labels = batch['labels'].numpy()
                    word_mask = labels != -100
                    labels[~word_mask] = self.tokenizer.pad_token_id
                    targets.extend(labels)

            results = {
                'predictions': predictions,
                'targets': targets
            }

            return results
        
    def postprocess_text(self, x_str):
        # Clean `bos` `eos` `pad` for cleaned text
        for to_remove_token in self.to_remove_token_list:
            x_str = x_str.replace(to_remove_token, '')

        return x_str.strip()    
    
    def compute_metrics(self, preds, labels, schema, decoding_format):
        # import pdb
        # pdb.set_trace()
        # preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded_preds = [self.postprocess_text(x) for x in decoded_preds]
        decoded_labels = [self.postprocess_text(x) for x in decoded_labels]
        
        result, well_formed_list = get_extract_metrics(
            pred_lns=decoded_preds,
            tgt_lns=decoded_labels,
            label_constraint=schema,
            decoding_format=decoding_format,
        )

        prediction_lens = [np.count_nonzero(pred.cpu() != self.tokenizer.pad_token_id) for pred in preds ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result, well_formed_list

    
    def compute_metrics_swig(self, preds, labels):
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded_preds = [self.postprocess_text(x) for x in decoded_preds]
        decoded_labels = [self.postprocess_text(x) for x in decoded_labels]

        result, well_formed_list= get_extract_metrics(
            pred_lns=decoded_preds,
            tgt_lns=decoded_labels,
            label_constraint=self.ace05_record_schema,
            decoding_format="swig",
        )

        prediction_lens = [np.count_nonzero(pred.cpu() != self.tokenizer.pad_token_id) for pred in preds ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result,well_formed_list
       

    def evaluate(self, loader, schema, decode_format, dump_path=None):
        results = self.predict(loader, dump_path)

        predictions = results['predictions']
        if dump_path is None:
            targets = results['targets']
            eval_results, well_formed_list= self.compute_metrics(predictions, targets, schema, decode_format)
   
            return eval_results, well_formed_list
    
    

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        print(args.gpu)
        # torch.cuda.set_device(f"cuda:{args.gpu}")
        # dist.init_process_group(backend='nccl')
        if args.use_deepspeed:
            deepspeed.init_distributed()
        else:
            torch.cuda.set_device(f"cuda:{args.gpu}")
            dist.init_process_group(backend='nccl')
            

    # use different type of inputs features
    trainer = Trainer(args,train=True)
    trainer.train()

if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count() # 8
    args.world_size = ngpus_per_node
    
    if args.local_rank in [0, -1]:
        print(args)
        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '': #'comment': ''
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M') #Feb20_13-10 %b 本地简化的月份名称 %d 月内中的一天
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'
        if args.run_name == "": #4tasks_hard_RN101_LMfull_bs500_image224_lr1e-4
            args.run_name = run_name
        if args.local_rank == -1:
            args.local_rank = args.cuda
    # if args.distributed:
    print(args.local_rank)
    main_worker(args.local_rank, args)
