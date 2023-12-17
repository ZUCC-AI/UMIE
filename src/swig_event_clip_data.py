from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = Path('/data2')
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
imsitu_dir = dataset_dir.joinpath('swig_event')

class SWiGFineTuneDataset(Dataset):
    def __init__(self, split='karpathy_train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train',collate_fn = None,dataset_prefix='swig_event',task='swig_event',tokenizer = None):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args
        self.tokenizer = tokenizer
        self.args.BUTD100 = False

        self.mode = mode

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)


        # if self.args.tokenizer is None:
        #     self.args.tokenizer = self.args.backbone

        # if 't5' in self.args.tokenizer:
        #     if self.args.use_vision:
        #         self.tokenizer = VLT5TokenizerFast.from_pretrained(
        #             args.backbone,
        #             # max_length=self.args.max_text_length,
        #             do_lower_case=self.args.do_lower_case)
        #     else:
        #         self.tokenizer = T5TokenizerFast.from_pretrained(
        #                 "/home/liqingyuan/VL_adapter/VL-T5/hf_models/uie-base-en")
        #             # max_length=self.args.max_text_length,
        #             # do_lower_case=self.args.do_lower_case)
        # elif 'bart' in self.args.tokenizer:
        #     self.tokenizer = BartTokenizer.from_pretrained(
        #         args.backbone,
        #         # max_length=self.args.max_text_length,
        #         do_lower_case=self.args.do_lower_case)

        #     additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1,-1, -1)] + \
        #             [f'<vis_extra_id_{i}>' for i in range(100-1,-1,-1)]
        #     special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
        #     num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        if self.args.oscar_tags:
            # Load VG Classes
            vg_classes = []
            with open(vg_dir.joinpath('objects_vocab.txt')) as f:
                for obj in f.readlines():
                    vg_classes.append(obj.split(',')[0].lower().strip())
            self.vg_classes = vg_classes

        data_info_path = imsitu_dir.joinpath(f'swig_{self.mode}_uie.json') ## 修改
        print(data_info_path)
        with open(data_info_path) as f:
            imsitu_data = json.load(f)

        split_rename = {
            'train': 'train',
            'dev': 'dev',
            'test': 'test'
        }

        n_images = 0
        data = []
        for datum in imsitu_data:
            img_id = datum['image_id'].split('.')[0]
            new_datum = {
                'img_id': img_id,
                'sent': datum['text'].strip(),
                'targets':datum['record'].strip(),
                'spot':datum['spot'],
                'asoc':datum['asoc'],
                'spot_asoc':datum['spot_asoc'],
                 "task": task
                # 'is_train': True,
            }
            n_images += 1
            data.append(new_datum)

        if self.verbose:
            print(f"{self.mode} has {n_images} images")
            print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        if isinstance(self.topk, float) and (0 < self.topk <= 1):
            used_samples = int(self.topk * len(data))
            data = random.sample(data, used_samples)
            if self.verbose:
                print(f"Use only {len(data)} data")

        elif self.topk > 0:
            data = data[:int(self.topk)]
            if self.verbose:
                print(f"Use only {len(data)} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        self.source_to_h5 = {}

        if self.args.max_n_boxes == 36:
            self.source_to_h5 = imsitu_dir.joinpath(f'picture/data_clip_{self.args.feature_type}_fc')

        self.txt_collate_fn = collate_fn
        self.task = task


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        ###### Image ######
        if self.args.use_vision:
            img_id = datum['img_id'].split(".h5")[0]
            out_dict['img_id'] = img_id


            h5_path = self.source_to_h5
            path = h5_path.joinpath(f"{img_id}.h5")
            with h5py.File(path, 'r') as f:
                
                attr_feat = f[f"{img_id}/features"][...]
                fc_feat = torch.tensor(f[f"{img_id}/fc_features"][...]).unsqueeze(0)
                regions_feat = torch.tensor(f[f"{img_id}/regions"][...])
                regions= torch.tensor(f[f"{img_id}/regions"][...])
                boxes  =f[f"{img_id}/bbox"][...]
                img_h = f[f'{img_id}/img_h'][()]
                img_w = f[f'{img_id}/img_w'][()]
                vis_feats = torch.cat((fc_feat,regions_feat),dim=0)
                boxes = torch.from_numpy(boxes)

                # print("vis_feats",vis_feats.shape)


                assert fc_feat[0].equal(vis_feats[0]),"error"
                # out_dict['vis_feats'] = feats # (L, D)
                out_dict['fc_feats'] = vis_feats # (L, D)
                out_dict["attr_feats"] = attr_feat
                out_dict["image"] = torch.tensor([])
                # boxes = torch.zeros(vis_feats.shape[0], 4) # (L, 4)

                out_dict['boxes'] = boxes
                out_dict['n_boxes'] = boxes.shape[0]
        ###### Text #####
        if self.args.no_prefix:
            input_text = ''
            input_ids = []
            
        else:
            if self.args.prefix is None:
                prefix = f'{self.args.prompt}'
            elif self.args.prefix == 'span':
                prefix = "span prediction:"
            elif self.args.prefix == 'denoise':
                prefix = "denoise text: <mask>"
            elif self.args.prefix == 'mask':
                if 'bart' in self.args.tokenizer:
                    prefix = "<mask>"
            input_tokens = [prefix]
            input_text = ' '.join(input_tokens)+ datum['sent']
            # if 't5' in self.args.tokenizer:
            input_ids = self.tokenizer.encode(
                    input_text,
                    max_length=self.args.max_text_length, truncation=True)


        out_dict['input_text'] = input_text
        out_dict['input_ids'] = input_ids
        out_dict['input_length'] = len(input_ids)
        out_dict['sample_prompt'] = [False] * len(out_dict['input_ids'])

        target = datum['targets'].strip()
        # if 't5' in self.args.tokenizer:
        target_ids = self.tokenizer.encode(target, max_length=192, truncation=True)

        
        
        out_dict['sent'] = datum['sent']
        out_dict['labels'] = target_ids
        out_dict['labels_length'] = len(target_ids)
        out_dict['spots'] = datum['spot']
        out_dict['asocs'] = []
        out_dict['spot_asoc'] = datum['spot_asoc']
        # sample_prompt=True for Finetune and Pretrain
        out_dict['sample_prompt'] = [True] * len(out_dict['input_ids'])
        out_dict['targets'] = datum['targets']
        out_dict['task'] = datum['task']
        out_dict["mode"] = self.mode

        return out_dict   
    
    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if self.args.no_prefix:
            assert input_ids.size() == (B, 0)

        if self.args.use_vision:
            # V_L = max(entry['n_boxes'] for entry in batch)
            V_L = max(entry['boxes'].shape[0] for entry in batch)
            # V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
            vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        # sentences = []

        targets = []
        img_ids = []
        img_paths = []
        input_text = []
        feature = self.txt_collate_fn(deepcopy(batch))
        # print(feature.keys())
        for i, entry in enumerate(batch):
            # print(entry.keys())
            # input_ids[i, :entry['input_length']] = torch.LongTensor(entry['input_ids'])

            if self.args.use_vision:
                n_boxes = entry['n_boxes']
                boxes[i] += entry['boxes']
                vis_feats[i] += entry['vis_feats']
                vis_attention_mask[i, :n_boxes] = 1
                img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])

            # if 'target_ids' in entry:
            #     target_ids[i, :entry['target_length']] = torch.LongTensor(entry['labels'])

            if 'input_text' in entry:
                input_text.append(entry['input_text'])

            # sentences.append(entry['sent'])

            if 'targets' in entry:
                targets.append(entry['targets'])


        # batch_entry['input_ids'] = input_ids
        # if 'label' in batch[0]:
        #     word_mask = target_ids != self.tokenizer.pad_token_id
        #     target_ids[~word_mask] = -100
        #     batch_entry['target_ids'] = target_ids

        if self.args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            batch_entry['vis_attention_mask'] = vis_attention_mask
            batch_entry['img_id'] = img_ids
            batch_entry['img_paths'] = img_paths

        # batch_entry['sent'] = sentences

        # batch_entry['input_text'] = input_text

        batch_entry['targets'] = targets
        batch_entry.update(feature)
        batch_entry['task'] = "swig_event"

        return batch_entry    
                           
def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1,collate_fn=None,tokenizer = None, dataset_prefix = "swig_event",task="swig_event"):

    # if 'mscoco' in split:
    verbose = (gpu == 0)

    dataset = SWiGFineTuneDataset(
        split,
        # raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode,collate_fn=collate_fn,
        tokenizer = tokenizer,  
        dataset_prefix = dataset_prefix,
        task = task
        )
    
    # indices = [x for x in range(8000)]
    # dataset = torch.utils.data.Subset(dataset, indices)
    # elif 'CC' in split:
    #     dataset = CCDataset(split, transform=transform, topk=topk)
    # import pdb
    # pdb.set_trace()
    if distributed and mode == 'train':
        # indices = [x for x in range(2000)]
        # dataset = torch.utils.data.Subset(dataset, indices)
        # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
        train_sampler = DistributedSampler(dataset)
        # train_sampler = RandomNonreplacmentSampler(dataset, dataset.n_iter)
    else:
        train_sampler = None
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, sampler=train_sampler,
            collate_fn = collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers,
            sampler=None,
            collate_fn=collate_fn,
            drop_last=False)

    # if verbose:
    #     loader.evaluator = COCOCaptionEvaluator()

    loader.task = task

    return loader, dataset



