# Gleason Grade

This project contains the code to train, test, and deploy semantic segmentation models.
The training data may be available by request.

If you find this code useful please cite:

```
bibtex
```

### Installation

Scripts may be run from the root directory of this project, or from any of the sub-directories.
Most of the scripts rely on our `tfmodels` package for CNN models, or `svs_reader` package for efficient and extensible reading of SVS format whole slide images.

```
pip install numpy opencv-contrib-python openslide-python tensorflow-gpu
git clone https://github.com/BioImageInformatics/gleason_grade
cd gleason_grade
git clone https://github.com/BioImageInformatics/svs_reader
git clone https://github.com/BioImageInformatics/tfmodels
```

### Deploying a trained model
Trained segmentation snapshots are available.
Retrained Hub modules are also available as standalone modules.

### Examples


### Notebooks
