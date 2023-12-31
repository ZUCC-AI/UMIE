#!/bin/bash
usage() {
  echo "Usage: ${0} [-t|--task] [-n|--node] [-m|--model] [-d|--datapath] [-p|--ports] [-b|--bs] [-e|--epoch] [-lr|--learning_rate]" 1>&2
  exit 1 
}
while [[ $# -gt 0 ]];do
  key=${1}
  case ${key} in
    -t|--task)
      task=${2}
      shift 2
      ;;
    -n|--node)
      node=${2}
      shift 2
      ;;
    -m|--model)
      model=${2}
      shift 2
      ;;
    -d|--datapath)
      datapath=${2}
      shift 2
      ;;
    -p|--ports)
      ports=${2}
      shift 2
      ;;
    -b|--bs)
      ports=${2}
      shift 2
      ;;
    -e|--epoch)
      epochs=${2}
      shift 2
      ;;
    -lr|--learning_rate)
      lr=${2}
      shift 2
      ;;
  esac
done
    
echo $model 
if [ $model == "flan-t5" ]
then
    folder_prefix="flan-t5"
    backbone="hf_models/flan-t5-base"
    batch_size=32
    lr=1e-4
elif [ $model == "flan-t5-large" ]
then
    folder_prefix="flan-t5-large"
    backbone="hf_models/flan-t5-large"
    batch_size=16
    lr=5e-5
    val="val"
elif [ $model == "flan-t5-xl" ]
then
    folder_prefix="flan-t5-xl"
    backbone="hf_models/flan-t5-xl"
    batch_size=16
    lr=5e-5
    val="val"
elif [ $model == "bart" ]
then
    folder_prefix="VLBart"
    backbone="facebook/bart-base"
    batch_size=128

elif [ $model == "uie" ]
then
    folder_prefix="uie-base-en"
    backbone="hf_models/uie-base-en"
    batch_size=32
    lr=1e-4
    val="val"
elif [ $model == "uie-large" ]
then
    folder_prefix="uie-large-en"
    backbone="hf_models/uie-large-en"
    batch_size=8
    lr=5e-5
    val="val"
fi
echo $model
echo $ports
echo $datapath
echo $taskdd
echo $folder_prefix
echo $backbone
feature=RN101
echo $node
bs=$(($batch_size*$node))
echo $bs
param="$bs-$lr-$backbone-$val_$task-node-${node}-acc-$rate-${data_prefix}"
echo $param
name="T5_UIE_TEST_MUTAL_ENTITY_$(date "+%Y%m%d%H%M%S")_$param"

echo $name
output=snap/${folder_prefix}_${task}/$name
mkdir -p $output
# python -m torch.distributed.launch \
#     --nproc_per_node=$node \
#     --master_port=$ports\
#     src/${task}.py \
    # --distributed --multiGPU \
# deepspeed  --num_gpus=2  src/${task}.py \
#         --deepspeed_config /home/liqingyuan/VL_adapter/VL-T5/scripts/image/ds_config.json\
TOKENIZERS_PARALLELISM=True PYTHONPATH=$PYTHONPATH:./src \
python src/${task}.py \
        --optim adamw \
        --warmup_ratio 0.06 \
        --clip_grad_norm 5 \
        --lr ${lr} \
        --epochs $epochs\
        --num_workers 1 \
        --backbone ${backbone} \
        --output $output ${@:2} \
        --num_beams 1\
        --batch_size ${batch_size} \
        --valid_batch_size ${batch_size} \
        --unfreeze_language_model \
        --feature ${feature} --n_boxes 36 \
        --feat_dim 512 \
        --image_size "(224,224)" \
        --tasks "event_ace05_arg,event_ace05_trigger,event_swig_arg,event_swig_trigger, mner_mner2015, relation_mnre_v1" \
        --eval_tasks "event_m2e2_arg, event_m2e2_trigger,event_m2e2_all, mner_mner2017, relation_mnre_v2" \
        --run_name $name \
        --max_text_length 256\
        --gen_max_length 128\
        --use_tasks_prompts \

        # --eval_tasks "event_m2e2_arg, event_m2e2_trigger, event_m2e2_two_stage" \
    # --prompt "Extract entity: "\
    # --use_separate_optimizer_for_visual \
    # --use_adam_for_visual \
    # --downsample \

    # --prompt "Extract their relation: "\
 #two stage  
    # --tasks "event_ace05_arg,event_ace05_trigger,event_swig_arg,event_swig_trigger, mner_mner2015, mner_mnersnap, mner_mner2017, relation_mnre_v2, relation_mnre_v1" \
    # --eval_tasks "event_m2e2_arg, event_m2e2_trigger, event_m2e2_two_stage" \

#    # --oscar_tags
# # echo $output
# output=/home/liqingyuan/VL_adapter/VL-T5/snap/uie-large-en_swig/T5_UIE_TEST_MUTAL_ENTITY_20230425222439_8-1e-4-hf_models/uie-large-en-swig-node-4
# datapath=../datasets/swig/
# python3 src/sel2record.py -p $output -g $datapath -v -d spotasoc -c src/closest_offset_en.yaml
# python3 src/eval_extraction.py -p $output  -g  $datapath -w -m ${eval_match_mode:-"normal"}
