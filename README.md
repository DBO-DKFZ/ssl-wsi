Self-Supervised Learning on WSI
-------------------------------

*Pretrained model checkpoints are [here](https://drive.google.com/drive/folders/1awqa9uNAtBFqfsdRMT4vB5TCG6QqrJKL?usp=sharing).* You can cite this repository in the sidebar if you use it for your research.

This repository contains PyTorch code to train on thousands of WSI with self-supervision, namely [DINO](https://arxiv.org/abs/2104.14294) and [PMSN](https://arxiv.org/abs/2210.07277). To achieve this we make use of [Lightning](https://github.com/Lightning-AI/pytorch-lightning), [Lightly](https://github.com/lightly-ai/lightly) and our own framework to handle WSI called [slide-tools](https://github.com/DBO-DKFZ/slide_tools).

This codebase was used to pretrain ViT-Tiny and ViT-Small models on roughly [10.000 diagnostic WSI from TCGA](https://portal.gdc.cancer.gov/repository?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_format%22%2C%22value%22%3A%5B%22svs%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22Diagnostic%20Slide%22%5D%7D%7D%5D%7D). You can find all available checkpoints [here](https://drive.google.com/drive/folders/1awqa9uNAtBFqfsdRMT4vB5TCG6QqrJKL?usp=sharing) (*method*-*data*-*tilesize*-*epochs*-*gpus*-*model*.pt) and here are the wandb reports for [DINO](https://api.wandb.ai/links/luzuku/ttqzvske) and [PMSN](https://api.wandb.ai/links/luzuku/d0yjqdv7).

 - *method*: DINO or PMSN
 - *data*: TCGA, ImageNet or 50/50 mixed
 - *tilesize*: 1, 2, 4 (multiple of native tilesize in the WSI for loading, afterwards transforms will randomcrop)
 - *epochs*: 300 for our experiments (takes roughly 24h)
 - *gpus*: 8 x A100 40GB
 - *model*: small or tiny

[This notebook](etc/export_model_from_ckpt.ipynb) shows you how to load them. We tried to adhere to the original implementations as much as possible especially concerning the scheduling of hyperparameters.

Getting started
---------------
Setup the environment:
```
git clone https://github.com/DBO-DKFZ/ssl-wsi.git
cd ssl-wsi
mamba env create -f environment.yml
mamba activate ssl-wsi
```

Modify *data.root*, *data.natural_root* in the [configs](cfg) and make your own CSV specifying the paths to the WSI and their annotations.

DINO
----
For a mixed 50/50 (data.natural_ratio 0.5), ViT-Tiny, native Tilesize:
```bash
python dino.py \
    --config cfg/dino.yaml \
    --logger.name mixed_s1_e300_g8_tiny \
    --trainer.max_epochs 300 \
    --data.batch_size_per_device 216 \
    --data.tile_size_multiple 1 \
    --trainer.precision 16-mixed \
    --data.num_workers 16 \
    --data.natural_ratio 0.5 \
    --logger.project dino_cluster_final \
    --model.arch tiny
```

For a pure WSI, ViT-Small, 2x Tilesize:
```bash
python dino.py \
    --config cfg/dino.yaml \
    --logger.name tcga_s2_e300_g8_small \
    --trainer.max_epochs 300 \
    --data.batch_size_per_device 216 \
    --data.tile_size_multiple 2 \
    --trainer.precision 16-mixed \
    --data.num_workers 16 \
    --data.natural_ratio 0 \
    --logger.project dino \
    --model.arch small
```

PMSN
----

For a pure ImageNet-1k (or wherever *data.natural_root* is pointing) ViT-Small: 
```bash
python pmsn.py \
    --config cfg/pmsn.yaml \
    --logger.name im1k_e300_g8_small \
    --trainer.max_epochs 300 \
    --data.batch_size_per_device 216 \
    --trainer.precision 16-mixed \
    --data.num_workers 16 \
    --logger.project pmsn \
    --model.arch small \
    --data.without_wsi true
```

Train on TCGA
-------------
We included automatically annotated tissue masks and a CSV containing all TCGA WSI we used in [TCGA.zip](etc/TCGA.zip). We used all diagnostic WSI that had 40x magnification available. `data.root` needs to point to a directory containing a folder for each TCGA project housing the corresponding WSI (have a look into **tcga_all_mag40.csv**). If you want to mix in (natural) images with an ImageNet-like folder structure then specify `data.natural_root`.

Vision Transformer
------------------
We included our own definition of a Vision Transformer in **vit.py** inspired by a few others (mentioned in the file). It is able to **accept other image sizes** than the one used during training like the original implementation. You can choose between **cosine** and **learnable position embeddings** and use flash attention v1 thanks to [lucidrains](https://github.com/lucidrains/vit-pytorch) and PyTorch 2.0.

Citations
---------
```bibtex
@misc{caron2021emerging,
    title   = {Emerging Properties in Self-Supervised Vision Transformers},
    author  = {Mathilde Caron and Hugo Touvron and Ishan Misra and Hervé Jégou and Julien Mairal and Piotr Bojanowski and Armand Joulin},
    year    = {2021},
    eprint  = {2104.14294},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
```bibtex
@misc{assran2022masked,
      title={Masked Siamese Networks for Label-Efficient Learning}, 
      author={Mahmoud Assran and Mathilde Caron and Ishan Misra and Piotr Bojanowski and Florian Bordes and Pascal Vincent and Armand Joulin and Michael Rabbat and Nicolas Ballas},
      year={2022},
      eprint={2204.07141},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
