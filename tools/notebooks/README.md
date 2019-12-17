The notebooks process data stored relative to this directory.
The devel directory tree is printed at the end of this file.

To facilitate sharing via Git, the notebook files (ipynb) are synced with Markdown (md) using [https://github.com/mwouts/jupytext](jupytext).

To install, simply run `pip install jupytext` inside a virtualenv or conda environment. 
Afterwards, run `jupytext --to notebook $markdown_file` to generate `ipynb` from `md`.

This project is set to ignore `ipynb` unless forcefully added with `git add -f`.
```
tools/
├── notebooks
├── seg.05x.densenet
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_25_22_19_23
│   └── snapshots
├── seg.05x.densenet_s
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_26_07_42_18
│   └── snapshots
├── seg.05x.fcn
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_26_12_26_45
│   └── snapshots
├── seg.05x.fcn_s
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_26_16_45_01
│   └── snapshots
├── seg.05x.unet
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_26_19_56_28
│   └── snapshots
├── seg.05x.unet_s
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_27_10_37_24
│   └── snapshots
├── seg.10x.densenet
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_25_23_44_50
│   └── snapshots
├── seg.10x.densenet_s
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_26_08_58_16
│   └── snapshots
├── seg.10x.fcn
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_26_13_28_59
│   └── snapshots
├── seg.10x.fcn_s
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_26_17_48_43
│   └── snapshots
├── seg.10x.unet
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_26_21_51_31
│   └── snapshots
├── seg.10x.unet_s
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_27_11_39_05
│   └── snapshots
├── seg.20x.densenet
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_26_03_43_53
│   └── snapshots
├── seg.20x.densenet_s
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_26_10_43_24
│   └── snapshots
├── seg.20x.fcn
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_26_15_07_40
│   └── snapshots
├── seg.20x.fcn_s
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_26_18_57_58
│   └── snapshots
├── seg.20x.unet
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_27_04_13_42
│   └── snapshots
├── seg.20x.unet_s
│   ├── debug
│   ├── inference
│   ├── logs
│   │   └── 2019_06_27_14_33_45
│   └── snapshots
└── tfhub_mobilenet
    ├── 01pct
    │   └── snapshot
    │       └── variables
    ├── 02pct
    │   └── snapshot
    │       └── variables
    ├── 05pct
    │   └── snapshot
    │       └── variables
    ├── 10pct
    │   └── snapshot
    │       └── variables
    ├── 25pct
    │   └── snapshot
    │       └── variables
    └── 75pct
        └── snapshot
            └── variables
```
