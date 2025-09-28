<h1 align="center">Generative Zoo</h1>

<p align="center">
  <a href="https://ps.is.mpg.de/person/ntomasz">Tomasz Niewiadomski</a>,
  <a href="https://ps.is.mpg.de/person/ayiannakidis">Anastasios Yiannakidis</a>,
  <a href="https://www.hanzcuevas.com/">Hanz Cuevas-Velasquez</a>,
  <a href="https://sites.google.com/view/soubhiksanyal/">Soubhik Sanyal</a>,<br/>
  <a href="https://ps.is.mpg.de/person/black">Michael J. Black</a>,
  <a href="https://imati.cnr.it/mypage.php?idk=PG-2">Silvia Zuffi</a>,
  <a href="https://kulits.github.io/">Peter Kulits</a>
</p>

<p align="center">
  <a href="https://genzoo.is.tue.mpg.de">Project Page</a> •
  <a href="https://genzoo-org-genzoo.hf.space/">Demo</a>
</p>

---

## 📖 Overview

Generative Zoo (GenZoo) provides a scalable pipeline for generating realistic 3D animal pose-and-shape training data.  
Models trained exclusively on GenZoo data achieve **state-of-the-art performance** on real-world 3D animal pose and shape estimation benchmarks.

<div align="center">
  <img src="https://download.is.tue.mpg.de/genzoo/teaser.jpg" width="100%">
</div>

---

## ⚙️ Installation

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repository:
```bash
   git clone https://github.com/x-tomasz/genzoo
   cd genzoo
```

3. Sync the uv environment:

```bash
uv sync --locked
```

---
## 📂 Model Weights & Configs

1. Register on the [project website](https://genzoo.is.tue.mpg.de), download the following files, and place them in `./data/`:
   - [`genzoo_1M.ckpt`](https://download.is.tue.mpg.de/download.php?domain=genzoo&resume=1&sfile=genzoo_1M.ckpt)
   - [`smal_plus.pkl`](https://download.is.tue.mpg.de/download.php?domain=genzoo&resume=1&sfile=smal_plus.pkl)
   - [`animal_attributes.npz`](https://download.is.tue.mpg.de/download.php?domain=genzoo&resume=1&sfile=animal_attributes.npz)
   - [`dog_poses.npz`](https://download.is.tue.mpg.de/download.php?domain=genzoo&resume=1&sfile=dog_poses.npz)

2. Download the following external dependencies into `./data/`:
   - [`submission_animal_realnvp_mask_pred_net_6000.pth`](https://github.com/silviazuffi/awol)
   - [`vitpose_backbone.pth`](https://github.com/shubham-goel/4D-Humans)

---

## 🚀 Inference

Run inference on cropped input images:
```bash
uv run python inference.py ./eg_input --render
```
---
## 🛠️ Data Generation Pipeline

Start the GenZoo pipeline notebook:
```bash
uv run jupyter notebook genzoo_pipeline.ipynb
```
On a headless server:
```bash
uv run jupyter notebook genzoo_pipeline.ipynb --no-browser --port=8888 --ip=0.0.0.0
```

---

## 🎓 Training

1. Download and extract Pascal VOC (≈2 GB) under `./data/` for synthetic augmentation (see [isarandi/synthetic-occlusion](https://github.com/isarandi/synthetic-occlusion)).

2. Download the GenZoo training datasets:
   - **v1**: weak perspective camera  
   - **v2**: full perspective camera  

*(Uncomment `## Full perspective` code in `hmr2/datasets/image_dataset.py` to use full perspective camera GT)*

3. Prepare WebDataset tars with your chosen paths:
```bash
uv run python prep_tars.py
```

4. Choose test set, and specify the path in `hmr2/configs/datasets_tar.yaml`, `COCO-VAL` field.
(Default is [`Animal3D` (real)](https://xujiacong.github.io/Animal3D) or [`GenZoo-Felidae` (synthetic)](https://download.is.tue.mpg.de/download.php?domain=genzoo&resume=1&sfile=GenZoo_Felidae.zip))

5. Run training:
```bash
uv run python train_genzoo.py exp_name=your_exp_name \
  data=mix_all experiment=hmr_vit_transformer trainer=gpu launcher=local
```
---
## 🙏 Acknowledgments

This project builds on and adapts parts of:
- [4D-Humans](https://github.com/shubham-goel/4D-Humans)
- [AWOL](https://github.com/silviazuffi/awol)  
- [SMAL](https://github.com/vchoutas/smal)
- [Synthetic Occlusion](https://github.com/isarandi/synthetic-occlusion)

Special thanks to all GenZoo co-authors for their contributions and support, and in particular to [Peter Kulits](https://kulits.github.io) for supervision and guidance.

---

## 🎤 ICCV 2025 Presentation 🌺🌴🌊

GenZoo will be presented as a **poster at ICCV 2025** in Hawaii:  

- **Poster ID:** [#788](https://iccv.thecvf.com/virtual/2025/poster/109)  
- **Session:** Tue, 21 Oct 2025 — 6:15 p.m. to 8:15 p.m. PDT  
- **Authors:** Tomasz Niewiadomski, Anastasios Yiannakidis, Hanz Cuevas Velasquez, Soubhik Sanyal, Michael J. Black, Silvia Zuffi, Peter Kulits  
 
--- 

## 📝 Citation

If you use GenZoo in your research, please cite:

```bibtex
@inproceedings{niewiadomski2025ICCV,
  author    = {Niewiadomski, Tomasz and Yiannakidis, Anastasios and Cuevas-Velasquez, Hanz and Sanyal, Soubhik and Black, Michael J. and Zuffi, Silvia and Kulits, Peter},
  title     = {Generative Zoo},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```
---

## 📜 License

We build off the [4D-Humans](https://github.com/shubham-goel/4D-Humans) codebase to perform our experiments. As such, inherited code falls under the original MIT license. Additions and modifications are released under a different license in accordance with institute requirements which has been prepended to [LICENSE.md](LICENSE.md). 