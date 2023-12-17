
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
from pprint import pprint

from param import parse_args

from caption_data import get_loader
from utils import load_state_dict, LossMeter, set_global_logging_level
import wandb
from pprint import pformat

from vis_encoder import get_vis_encoder
from transformers.models.t5.modeling_t5 import T5LayerNorm
import modeling_t5
import modeling_bart
from clip.model import VisualAdapter

set_global_logging_level(logging.ERROR, ["transformers"])

proj_dir = Path(__file__).resolve().parent.parent


_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None,  record_schema=None, decoding_format=None, decoding_type_schema=None,model_class=None, train=True, source_prefix=None):
        super().__init__(args)
        # import pdb
        # pdb.set_trace()
        self.record_schema = record_schema
        self.decoding_format = decoding_format
        self.decoding_type_schema = decoding_type_schema

        self.wandb_initialized = False


        model_kwargs = {}

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        
        print(self.tokenizer)

        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1,-1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1 , -1)]
                        
                for special_token in [constants.type_start, constants.type_end, constants.span_start, constants.spot_prompt, constants.asoc_prompt]:
                    if special_token not in self.tokenizer.get_vocab():
                        additional_special_tokens += [special_token]
                        
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
                        
               
                
                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])
                self.model = self.create_model(model_class, config, **model_kwargs)
                self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)

        if 't5' in self.args.tokenizer or 'uie' in self.args.tokenizer : 
            to_add_special_token = list()
            for special_token in [constants.type_start, constants.type_end, constants.span_start, constants.spot_prompt, constants.asoc_prompt]:
                if special_token not in self.tokenizer.get_vocab():
                    to_add_special_token += [special_token]
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": to_add_special_token})
            self.model = self.create_model(model_class, config, **model_kwargs)
            self.model.resize_token_embeddings(len(self.tokenizer))

        
        

        self.constraint_decoder = get_constraint_decoder(tokenizer=self.tokenizer,
                                                    type_schema=self.record_schema,
                                                    decoding_schema="spotasoc",
                                                    source_prefix=None,
                                                    task_name="entity")
        self.spot_asoc_nosier = SpotAsocNoiser(
                    spot_noise_ratio=0.1,
                    asoc_noise_ratio=0.1,
                    null_span='<extra_id_6>',
                )

        self.data_collator = DataCollatorForMetaSeq2Seq(
            tokenizer = self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            max_length=256,
            max_prefix_length=-1,
            max_target_length=self.args.gen_max_length,
            negative_sampler=DynamicSSIGenerator(
                tokenizer=self.tokenizer,
                schema=self.record_schema,
                positive_rate=1,
                negative=-1,
                ordered_prompt=False,
            ),
            spot_asoc_nosier=self.spot_asoc_nosier,
            decoding_format='spotasoc',
        )
        from mner_clip_data import get_loader

        self.train_loader = get_loader(
            args,
            split=args.train, mode='train', batch_size=args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=args.num_workers,
            topk=args.train_topk,
            collate_fn = self.data_collator,
            tokenizer = self.tokenizer
        )
        
        if args.gpu == 0:
            if args.valid_batch_size is not None:
                valid_batch_size = args.valid_batch_size
            else:
                valid_batch_size = args.batch_size
                
            print(f'Building val loader at GPU {args.gpu}')

            self.val_loader = get_loader(
                args,
                split=args.valid, mode='val', batch_size=128,
                distributed=False, gpu=args.gpu,
                workers=4,
                topk=args.valid_topk,
                collate_fn = self.data_collator,
                tokenizer = self.tokenizer

            )
            print('# len val loader:', len(self.val_loader))

            print(f'Building test loader at GPU {args.gpu}')
            
            self.test_loader = get_loader(
                args,
                split=args.test, mode='test', batch_size=128,
                distributed=False, gpu=args.gpu,
                workers=4,
                topk=args.valid_topk,
                collate_fn = self.data_collator,
                tokenizer = self.tokenizer

            )
            print('# len text loader:', len(self.val_loader))

        self.to_remove_token_list = list()
        if self.tokenizer.bos_token:
            self.to_remove_token_list += [self.tokenizer.bos_token]
        if self.tokenizer.eos_token:
            self.to_remove_token_list += [self.tokenizer.eos_token]
        if self.tokenizer.pad_token:
            self.to_remove_token_list += [self.tokenizer.pad_token]
     

        self.model.tokenizer = self.tokenizer

        if self.include_vis_encoder:
            # train vision encoder end-to-end
            vis_encoder_type = self.args.feature_type.split("_")[-1]

            if self.args.use_vis_adapter:
                self.vis_encoder = get_vis_encoder(
                    backbone=vis_encoder_type, 
                    image_size=eval(self.args.image_size)[0],
                    adapter_type=self.args.vis_adapter_type,
                    reduction_factor=self.args.vis_reduction_factor,
                )
            else:
                self.vis_encoder = get_vis_encoder(
                    backbone=vis_encoder_type, 
                    image_size=eval(self.args.image_size)[0],
                    adapter_type=None,
                )

            self.model.vis_encoder = self.vis_encoder

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
        self.model = self.model.to(args.gpu)

        self.freeze_whole_model() # freeze whole parameters first
        self.unfreeze_parameters() # unfreeze selected parameters

        self.percent_updated_parameters = self.print_trainable_params_percentage(self.model)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def train(self):
        if self.verbose:
            loss_meter = LossMeter()
            best_valid = 0.
            best_epoch = 0

            if not self.wandb_initialized:

                if 't5' in self.args.backbone:
                    project_name = "VLT5_MNER"
                elif 'bart' in self.args.backbone:
                    project_name = "VLBart_MNER"
                elif 'uie' in self.args.backbone:
                    project_name = "VLUIE_MNER"

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
        epochs = self.args.epochs

        for epoch in range(epochs):

            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            self.partial_eval()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=120)

            epoch_results = {
                'loss': 0.,

            }

            for step_i, batch in enumerate(self.train_loader):

                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()


                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(
                            self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                update = True
                if self.args.gradient_accumulation_steps > 1:
                    if step_i == 0:
                        update = False
                    elif step_i % self.args.gradient_accumulation_steps == 0 or step_i == len(self.train_loader) - 1:
                        update = True
                    else:
                        update = False

                if update:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    # self.model.zero_grad()
                    for param in self.model.parameters():
                        param.grad = None
                    global_step += 1

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                if self.verbose:
                    loss_meter.update(loss.item())
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} | Steps {global_step}'
                    desc_str += f' | Loss {loss_meter.val:4f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.args.distributed:
                dist.barrier()

            if self.verbose:
                pbar.close()
                val_results, predictions= self.evaluate(self.val_loader)
                test_results,test_predictions= self.evaluate(self.test_loader)

                # wandb.log(val_results, step=epoch)
                print(val_results)
                print(test_results)
                overall_F1 = test_results['overall-F1']
                # overall_F1 = val_results['spot-R']
                if overall_F1 > best_valid or epoch == 0:
                    best_valid = overall_F1
                    best_epoch = epoch
                    self.save("BEST")

                    output_eval_file = os.path.join(self.args.output, "eval_results_seq2seq.txt")
                    with open(output_eval_file, "w") as writer:
                        for key, value in sorted(results.items()):
                            writer.write(f"{key} = {value}\n")
                        
                    eval_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    
                    eval_preds = [self.postprocess_text(pred) for pred in eval_preds]
                    output_test_preds_file = os.path.join(self.args.output, "eval_preds_seq2seq.txt")
                    with open(output_test_preds_file, "w") as writer:
                        writer.write("\n".join(eval_preds))
                    

                log_str = ''

                log_str += pformat(val_results)
                log_str += "\nEpoch %d: Valid  overall-F1 %0.4f" % (epoch,  overall_F1)
                log_str += "\nEpoch %d: Best  overall-F1 %0.4f\n" % (best_epoch, best_valid)

                wandb_log_dict = {}
                wandb_log_dict['Train/Loss'] = epoch_results['loss'] / len(self.train_loader)

                for score_name, score in val_results.items():
                    wandb_log_dict[f'Valid/{score_name}'] = score

                wandb_log_dict[f'Valid/best_epoch'] = best_epoch

                wandb.log(wandb_log_dict, step=epoch)
                

            # #     print(log_str)

            if self.args.distributed:
                dist.barrier()

        if self.verbose:
            self.save("LAST")

            # Test Set
            best_path = os.path.join(self.args.output, 'BEST')
            self.load(best_path)

            wandb.save(best_path, base_path=self.args.output)
            output_test_result_file = os.path.join(self.args.output, "test_results_seq2seq.txt")

            print(f'\nUploaded checkpoint {best_epoch}', best_path)
            
            test_results, predictions = self.evaluate(self.test_loader)
            
            with open(output_test_result_file, "w") as writer:
                # logger.info("***** Test results *****")
                for key, value in sorted(test_results.items()):
                    # logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
                    
            test_preds = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
            
            test_preds = [self.postprocess_text(pred) for pred in test_preds]
            
            output_test_preds_file = os.path.join(self.args.output, "test_preds_seq2seq.txt")
            with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(test_preds))

            wandb_log_dict = {}
            for score_name, score in test_results.items():
                wandb_log_dict[f'Test/{score_name}'] = score
            wandb.log(wandb_log_dict, step=epoch)

            log_str = 'Test set results\n'
            log_str += pformat(test_results)

            print(log_str)

        if self.args.distributed:
            dist.barrier()

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
                    # print(labels)
                    # import pdb
                    # pdb.set_trace()
                    word_mask = labels != -100
                    labels[~word_mask] = self.tokenizer.pad_token_id
                    # print(labels)
                    targets.extend(labels)
                    # print(targets)

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
    
    def compute_metrics(self, preds, labels):
        # import pdb
        # pdb.set_trace()
        # preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
        # labels = np.where(np.array(labels) != -100, labels, self.tokenizer.pad_token_id)
        # word_mask = np.array(labels) != -100
        # labels[~word_mask] = self.tokenizer.pad_token_id
        # print(labels)

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded_preds = [self.postprocess_text(x) for x in decoded_preds]
        decoded_labels = [self.postprocess_text(x) for x in decoded_labels]

        # print(decoded_preds)
        # print(decoded_labels)
        # import pdb
        # pdb.set_trace()
        result = get_extract_metrics(
            pred_lns=decoded_preds,
            tgt_lns=decoded_labels,
            label_constraint=self.record_schema,
            decoding_format="entity",
        )

        prediction_lens = [np.count_nonzero(pred.cpu() != self.tokenizer.pad_token_id) for pred in preds ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    def evaluate(self, loader, dump_path=None):
        # evaluator = loader.evaluator
        results = self.predict(loader, dump_path)

        predictions = results['predictions']
        # print(predictions)
        if dump_path is None:
            targets = results['targets']
            eval_results = self.compute_metrics(predictions, targets)
            # eval_results = []
            return eval_results,predictions
        

        

    

    
    @staticmethod
    def oracle_score(loader):
        evaluator = loader.evaluator
        quesid2ans = {}
        for i, batch in enumerate(loader):

            ques_id = batch['question_ids']
            label = batch['targets']

            _, label = label.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = loader.dataset.raw_dataset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    # use different type of inputs features
    if args.feature_type == "butd":
        from caption_data import get_loader
    elif args.feature_type.startswith("raw"):
        feature_type = args.feature_type.split("_")[-1]

        if args.vis_pooling_output:
            feat_dim_dict = {
                "RN50": 1024,
                "RN101": 512,
                "RN50x4": 640,
            }
        else:
            feat_dim_dict = {
                "RN50": 2048,
                "RN101": 2048,
                "RN50x4": 2560,
                "ViT": 768
            }
        args.feat_dim = feat_dim_dict[feature_type]

        from caption_raw_data import get_loader
    else:
        feat_dim_dict = {
            "RN50": 2048,
            "RN101": 2048,
            "RN50x4": 2560,
            "ViT": 768
        }
        args.feat_dim = feat_dim_dict[args.feature_type]
        from mner_clip_data import get_loader
    record_schema = RecordSchema.read_from_file(f"/home/data2/datasets/MNER/anno_ner_{args.data_prefix}/record.schema")
    decoding_format = "tree"
    decoding_type_schema = "mner"
    
    from mner_model import VLT5MNER, VLBartMNER

    if 't5' in args.backbone or 'uie' in args.backbone  :
        model_class = VLT5MNER
    elif 'bart' in args.backbone:
        model_class = VLBartMNER

    # print(f'Building train loader at GPU {gpu}')
    # train_loader = get_loader(
    #     args,
    #     split=args.train, mode='train', batch_size=args.batch_size,
    #     distributed=args.distributed, gpu=args.gpu,
    #     workers=args.num_workers,
    #     topk=args.train_topk,
    # )
    # if gpu == 0:
    #     if args.valid_batch_size is not None:
    #         valid_batch_size = args.valid_batch_size
    #     else:
    #         valid_batch_size = args.batch_size
    #     print(f'Building val loader at GPU {gpu}')
    #     val_loader = get_loader(
    #         args,
    #         split=args.valid, mode='val', batch_size=valid_batch_size,
    #         distributed=False, gpu=args.gpu,
    #         workers=4,
    #         topk=args.valid_topk,
    #     )
    #     print('# len val loader:', len(val_loader))

    #     print(f'Building test loader at GPU {gpu}')
    #     test_loader = get_loader(
    #         args,
    #         split=args.test, mode='test', batch_size=valid_batch_size,
    #         distributed=False, gpu=args.gpu,
    #         workers=4,
    #         topk=args.valid_topk,
    #     )
    # else: 
    train_loader = None
    val_loader = None
    test_loader = None

    trainer = Trainer(args, train_loader, val_loader, test_loader, record_schema,decoding_format, decoding_type_schema,model_class, train=True)
    trainer.train()



if __name__ == "__main__":

    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    print(type(args))
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        if args.run_name == "":
            args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
