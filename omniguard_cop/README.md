<div align="center">
<h1> Copyright Watermark Model Training </h1>
</div>

The OmniGuard is composed of two parts: Anti-manipulation watermark model and Copyright watermark model. This folder stores the scripts used to train the Copyright watermark model.

Before running the code, you need to install the required packages:
```
pip install -r requirements.txt
```
and configure [accelerate](https://github.com/huggingface/accelerate) by running the following command:
```
accelerate config
```


## Stage 1: Pretraining
This training stage is used to pretrain the model to equip it with robustness towards common degradations.
```
accelerate launch pretrain_ddp.py --train-root /path/to/train --val-root /path/to/val --batch-size 32
```

## Stage 2: Finetuning
This training stage is used to finetune the pretrained model to enhance its capability to endure VAE regenerations. You are required to provide the path of the pretrained model.
```
accelerate launch finetune_ddp.py --train-root /path/to/train --val-root /path/to/val --pretrained /path/to/pretrained --batch-size 8
```
