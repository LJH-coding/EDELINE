# EDELINE: Enhancing Memory in Diffusion-based World Models via Linear-Time Sequence Modeling

**TL;DR** EDELINE combines diffusion models with state space models to create a world model for reinforcement learning that overcomes memory limitations in previous approaches.

<div align='center'>
<img src='media/edeline.gif' width="100%"/>
</div>

Quick install using [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/):

>```bash
>git clone https://github.com/LJH-coding/EDELINE.git
>cd EDELINE
>conda create -n edeline python=3.10
>conda activate edeline
>pip install -r requirements.txt
>```

**Warning**: Atari ROMs will be downloaded with the dependencies, which means that you acknowledge that you have the license to use them.

<a name="quick_links"></a>
## Quick Links

- [Launch a training run](#launch)
- [Configuration](#configuration)
- [Run folder structure](#structure)
- [Citation](#citation)
- [Credits](#credits)

<a name="launch"></a>
## [⬆️](#quick_links) Launch a training run

To train with the hyperparameters used in the paper, launch:
```bash
python src/main.py env.train.id=BreakoutNoFrameskip-v4
```

This creates a new folder for your run, located in `outputs/YYYY-MM-DD/hh-mm-ss/`.

To resume a run that crashed, navigate to the fun folder and launch:

```bash
./scripts/resume.sh
```

<a name="configuration"></a>
## [⬆️](#quick_links) Configuration

We use [Hydra](https://github.com/facebookresearch/hydra) for configuration management.

All configuration files are located in the `config` folder:

- `config/trainer.yaml`: main configuration file.
- `config/agent/default.yaml`: architecture hyperparameters.
- `config/env/atari.yaml`: environment hyperparameters.

You can turn on logging to [weights & biases](https://wandb.ai) in the `wandb` section of `config/trainer.yaml`.

Set `training.model_free=true` in the file `config/trainer.yaml` to "unplug" the world model and perform standard model-free reinforcement learning.

<a name="structure"></a>
## [⬆️](#quick_links) Run folder structure

Each new run is located at `outputs/YYYY-MM-DD/hh-mm-ss/`. This folder is structured as follows:

```txt
outputs/YYYY-MM-DD/hh-mm-ss/
│
└─── checkpoints
│   │   state.pt  # full training state
│   │
│   └─── agent_versions
│       │   ...
│       │   agent_epoch_00999.pt
│       │   agent_epoch_01000.pt  # agent weights only
│
└─── config
│   |   trainer.yaml
|
└─── dataset
│   │
│   └─── train
│   |   │   info.pt
│   |   │   ...
|   |
│   └─── test
│       │   info.pt
│       │   ...
│
└─── scripts
│   │   resume.sh
|   |   ...
|
└─── src
|   |   main.py
|   |   ...
|
└─── wandb
    |   ...
```

<a name="citation"></a>
## [⬆️](#quick-links) Citation

```text
@inproceedings{
  lee2025edeline,
  title={{EDELINE}: Enhancing Memory in Diffusion-based World Models via Linear-Time Sequence Modeling},
  author={Jia-Hua Lee and Bor-Jiun Lin and Wei-Fang Sun and Chun-Yi Lee},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=ph1V6n7BSv}
}
```

<a name="credits"></a>
## [⬆️](#quick_links) Credits

- https://github.com/eloialonso/diamond
- https://github.com/alxndrTL/mamba.py
- https://github.com/crowsonkb/k-diffusion/
- https://github.com/huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub)
- [https://github.com/google-research/rliable](https://github.com/google-research/rliable)
- [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
