# UMIE
Code and model for AAAI 2024: UMIE: Unified Multimodal Information Extraction with Instruction Tuning


# UMIE: Unified Multimodal Information Extraction with Instruction Tuning

## Model Architecture

The overall architecture of our hierarchical modality fusion network.

![架构图-2-5.drawio](models/model.png)

## Datasets

![path](datasets/path.png)


## Data Preprocess

**Vision**

To extract visual object images, we first use the NLTK parser to extract noun phrases from the text and apply the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. Detailed steps are as follows:

1. Using the NLTK parser (or Spacy, textblob) to extract noun phrases from the text.
2. Applying the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. Taking the twitter2015 dataset as an example, the extracted objects are stored in `twitter2015_images.h5`. The images of the object obey the following naming format: `imgname_pred.png`, where `imgname` is the name of the raw image corresponding to the object, `num` is the number of the object predicted by the toolkit.

The detected objects and the dictionary of the correspondence between the raw images and the objects are available in our data links.

![VOA_EN_NW_2015.10.21.3017239_4](datasets/event.png)

**Text**

```bash
bash data_processing/run_data_generation.bash
```

Exapmle:

**Entity**

```json
{
 "text": "@USER Kyrie plays with @USER HTTPURL",
 "label": "person, Kyrie", 
 "image_id": "149.jpg"
}
```

**Relation**

```json
{
	"text": "Do Ryan Reynolds and Blake Lively ever take a bad photo ? 😍",
	"label": "Ryan Reynolds <spot> couple <spot> Blake Lively", 
	"image_id": "O_1311.jpg"
}
```

**Event Trigger**

```json
{
    "text":"Smoke rises over the Syrian city of Kobani , following a US led coalition airstrike, seen from outside Suruc",
    "label": "attack, airstrike",
    "image_id": "VOA_EN_NW_2015.10.21.3017239_4.jpg"
}
```

**Event Argument**

```json
{
    "text":"Smoke rises over the Syrian city of Kobani , following a US led coalition airstrike, seen from outside Suruc",
    "label": "attack <spot> attacker, coalition <spot> Target, O1",
  	"O1": [1, 190, 508, 353],
    "image_id": "VOA_EN_NW_2015.10.21.3017239_4.jpg"
}
```

## Data Download

- Twitter2015 & Twitter2017

  The text data follows the conll format. You can download the Twitter2015 data via this [link](https://drive.google.com/file/d/1qAWrV9IaiBadICFb7mAreXy3llao_teZ/view?usp=sharing) and download the Twitter2017 data via this [link](https://drive.google.com/file/d/1ogfbn-XEYtk9GpUECq1-IwzINnhKGJqy/view?usp=sharing). Please place them in `data/NNER`.
- MNRE

  The MNRE dataset comes from [MEGA](https://github.com/thecharm/MNRE), many thanks.

  MEE
- The MEE dataset comes from [MEE](https://github.com/limanling/m2e2), many thanks.
- SWiG

  The SWiG dataset comes from [SWiG](https://github.com/thecharm/MNRE), many thanks.

## Requirements

To run the codes, you need to install the requirements:

```bash
pip install -r requirements.txt
```

## Train

```bash
bash -x scripts/image/full_finetuning.sh -p 1 --task ie_multitask --model flan-t5 --ports 26754 --epoch 30  --lr 1e-4
```

## Test

```bash
bash -x scripts/image/test_eval.sh -p 1 --task ie_multitask --model flan-t5 --ports 26768 
```