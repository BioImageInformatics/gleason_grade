# Gleason Grading

This project contains the code to train, test, and deploy semantic segmentation models.
The training data may be available by request.

If you find this project useful please cite:

```

//bibtex

in progress

```

### Installation

Scripts may be run from the root directory of this project, or from any of the sub-directories.
Most of the scripts rely on our [`tfmodels`](https://github.com/BioImageInformatics/tfmodels) package for CNN models, or [`svs_reader`](https://github.com/BioImageInformatics/svs_reader) package for interfacing image pyramids stored in Aperio's SVS format.
At this time there are no plans to move any of these packages into pypi or conda.

```
pip install numpy opencv-contrib-python openslide-python tensorflow-gpu pandas
git clone https://github.com/BioImageInformatics/gleason_grade
cd gleason_grade
git clone https://github.com/BioImageInformatics/svs_reader ./svs_reader
git clone https://github.com/BioImageInformatics/tfmodels ./tfmodels
```

Tested on Ubuntu.

### Directory structure
---
```
gleason_grade
|__ data
    |__ train_jpg (1)
    |__ train_mask (2)
    |__ save_tfrecord.py (3)
    |__ misc utility scrips
|__ densenet (4)
    |__ densenet.py (5)
    |__ train.py (6)
    |__ test.py (7)
    ...
|__ densenet_small
    ...
|__ fcn8s
...
|__ notebooks (8)
|__ tfhub (9)
    |__ create_tfhub_training.py (10)
    |__ retrain.py (11)
    |__ deploy_retrained.py (12)
    |__ test_retrained.py (13)
    |__ run_retrain.sh (14)
    |__ run_deploy.sh (15)
|__ classifiers
    |__ W.I.P.

```
1. A directory with source training images
2. A directory with source training masks, **name matched to (1)**
3. Utility for translating (1) and (2) into `tfrecord` format for training
4. Model directory. Each model gets its own directory for organizing snapshots and results.
5. The model definition file. This extends one of the base classes in `tfmodels`
6. Training script. Each model directory has a copy.
7. Testing script. Each model directory has a copy.
8. A set of jupyter notebooks for running various experiments, and collecting results. Notably, `colorize_numpy.ipynb` will read the output files in a given directory and produce a color-coded png based on a given color scheme.
9. [Tensorflow HUB](https://www.tensorflow.org/hub/) experiments.
10. Translate images in (1) and (2) into labelled tiles for classifier trainig
11. The retraining script from `tensorflow/examples/image_retraining`
12. Script to apply retrained Hub modules to SVS files
13. Run a test on retrained Hub classifiers
14. Utility script to hold options for retraining
15. Utility script to hold options for deploy
