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
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')
coco_img_dir = coco_dir.joinpath('images/')
coco_feature_dir = coco_dir.joinpath('features')
imsitu_dir = dataset_dir.joinpath('ACE05')

class ACE05FineTuneDataset(Dataset):
    def __init__(self, split='karpathy_train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train',collate_fn = None,dataset_prefix='2015',task="ace05-evt",tokenizer = None):
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

        if self.args.oscar_tags:
            # Load VG Classes
            vg_classes = []
            with open(vg_dir.joinpath('objects_vocab.txt')) as f:
                for obj in f.readlines():
                    vg_classes.append(obj.split(',')[0].lower().strip())
            self.vg_classes = vg_classes

        data_info_path = imsitu_dir.joinpath(f'ace_{self.mode}_uie.json') ## 修改
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
            img_id = datum['image_id'].split('.jpg')[0]
            img_id = img_id.split('.png')[0]
            img_id = img_id.split('.JPG')[0]

            new_datum = {
                'img_id': img_id,
                'sent': datum['text'].strip(),
                'targets':datum['record'].strip(),
                'spot':datum['spot'],
                'asoc':datum['asoc'],
                'spot_asoc':datum['spot_asoc'],
                 "task": task,
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
            self.source_to_h5 = "/data2/datasets/voa/voa_img/data_clip_RN101_fc"

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
            img_id = datum['img_id']
            out_dict['img_id'] = img_id
             
            h5_path = self.source_to_h5
            from os import path
            path = path.join(h5_path,f"{img_id}.h5")
            with h5py.File(path, 'r') as f:
                
                attr_feat = f[f"{img_id}/features"][...]
                fc_feat = torch.tensor(f[f"{img_id}/fc_features"][...]).unsqueeze(0)
                regions_feat = torch.tensor(f[f"{img_id}/regions"][...])
                regions= torch.tensor(f[f"{img_id}/regions"][...])
                boxes  =f[f"{img_id}/bbox"][...]
                img_h = f[f'{img_id}/img_h'][()]
                img_w = f[f'{img_id}/img_w'][()]
                boxes = torch.from_numpy(boxes)
                vis_feats = torch.cat((fc_feat,regions_feat),dim=0)


                assert fc_feat[0].equal(vis_feats[0]),"error"
                out_dict['fc_feats'] = vis_feats # (L, D)
                out_dict["attr_feats"] = attr_feat
                out_dict["image"] = torch.tensor([])
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
            input_ids = self.tokenizer.encode(
                    input_text,
                    max_length=self.args.max_text_length, truncation=True)


        out_dict['input_text'] = input_text
        out_dict['input_ids'] = input_ids
        # print(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['sample_prompt'] = [False] * len(out_dict['input_ids'])

        target = datum['targets'].strip()
        target_ids = self.tokenizer.encode(target, max_length=192, truncation=True)

        
        out_dict['sent'] = datum['sent']
        out_dict['labels'] = target_ids
        out_dict['labels_length'] = len(target_ids)
        out_dict['spots'] = datum['spot']
        out_dict['asocs'] = datum['asoc']
        out_dict['spot_asoc'] = datum['spot_asoc']
        # sample_prompt=True for Finetune and Pretrain
        out_dict['sample_prompt'] = [True] * len(out_dict['input_ids'])
        out_dict['targets'] = datum['targets']
        out_dict['task'] = datum['task']
        out_dict["mode"] = self.mode
        return out_dict   
      
                           
def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1,collate_fn=None,tokenizer = None, dataset_prefix = "swig", task="ace05-evt"):

    # if 'mscoco' in split:
    verbose = (gpu == 0)

    dataset = ACE05FineTuneDataset(
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
    if distributed and mode == 'train':
   # The code `# indices = [x for x in range(2000)]` and `# dataset = torch.utils.data.Subset(dataset,
   # indices)` are commented out, which means they are not currently being used in the code.
        # indices = [x for x in range(2000)]
        # dataset = torch.utils.data.Subset(dataset, indices)
        # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
        train_sampler = DistributedSampler(dataset)
    else:
        train_sampler = None
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=False, sampler=train_sampler,
            collate_fn = collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=workers, 
            pin_memory=False,
            sampler=None,
            collate_fn=collate_fn,
            drop_last=False)

    loader.task = task

    return loader, dataset



