## Causal Diffusion Transformer for Generative Modeling

![samples](figures/vis.png)
![samples](figures/edit.png)


This repo provides the official implementation for our paper
> **[Causal Diffusion Transformers for Generative Modeling](https://arxiv.org/)**<br>
> Chaorui Deng, [Deyao Zhu](https://tsutikgiau.github.io/), [Kunchang Li](https://andy1621.github.io/), Shi Guang, [Haoqi Fan](https://haoqifan.github.io/)
> <br>Bytedance Research<br>

### Setup
Install the dependencies:
```bash
git clone https://github.com/causalfusion/causalfusion.git
pip install -U torch==2.5.1 torchvision==0.20.1 transformers==4.46.2
```
Download pretrained VAE from [MAR](https://github.com/LTH14/mar?tab=readme-ov-file#installation).



### Training

Training CausalFusion-XL on 8 GPUs with a batch size of 256:

```bash
torchrun --nnodes=1 --nproc_per_node=8 train.py --data-path=$PATH_TO_IMAGENET_TRAIN_DIR --tokenizer-path=$PATH_TO_MAR_VAE --results-dir=$PATH_TO_RESULTS_DIR --model=CausalFusion-XL --global-batch-size=256 --ckpt-every=50000 --lr=1e-4 --distributed --grad-checkpoint
```

### Sampling
Download pretrained [CausalFusion-XL](https://drive.google.com/file/d/1Z0xly7gaXASJnbeWNMjQBwbZdUnLFLu1/view?usp=sharing).

Sampling 10,000 images on 8 GPUs with CFG scale of 4.0:
```bash
torchrun --nnodes=1 --nproc_per_node=8 sample.py --distributed --model=CausalFusion-L --tokenizer-path=$PATH_TO_MAR_VAE --num-fid-samples=10000 --ckpt=$PATH_TO_PRETRAINED_CKPT --sample-dir=$PATH_TO_SAMPLE_DIR --cfg-scale=4.0
```


### Evaluation
See the instrutions in [ADM](https://github.com/openai/guided-diffusion/tree/main/evaluations) for evaluation.


### BibTex

```latex
@article{deng2024causalfusion,
  title={Causal Diffusion Transformers for Generative Modeling},
  author={Chaorui Deng, Deyao Zhu, Kunchang Li, Shi Guang, Haoqi Fan},
  year={2024},
  journal={arXiv preprint arXiv:},
}
```


### Acknowledgments

This codebase borrows from [DiT](https://github.com/facebookresearch/DiT) and [ADM](https://github.com/openai/guided-diffusion), thanks for their great works!

