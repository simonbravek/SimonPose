[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# SimonPOSE
## How to run

After you are all set up, you can should run the projects from the repo root so that you could use the ```config.py``` with all the paths. You can alter this file if you for any reason decide not to stick with our file structure. The projects are treated as separate python packages and so you should run them from the project root with the command

```
python3 projects.[project name]
```

## Setup
### On CTU cluster
If you have access to the CTU university servers, there is a setup guide for the repo. 
#### Compute
Look at the GPU occupation on the cluster: 
```
nvidia-smi
```
and set you environment variable so that you see only the free device you want to use for the best performance. You can do that by prepanding this when you run the script: 
```
CUDA_VISIBLE_DEVICES=8 python3 projects.plane_distance      # Or whatever number is free
```
#### Environment
The heavy modules are already on the system and should simply be loaded through

```
ml PyTorch/2.4.0-foss-2023b-CUDA-12.4.0 torchvision/0.19.0-foss-2023b-CUDA-12.4.0 OpenCV/4.10.0-foss-2023b-CUDA-12.4.0-contrib Albumentations/1.4.4-foss-2023b-CUDA-12.4.0 pycocotools/2.0,7-foss-2023b matplotlib/3.8.2-gfbf-2023b
```

and create an environment with the following packages. I have simply done it using the native python environment manager inside of the project's directory. You can also use ```conda``` for the management. 
```
python3.11 -m venv .venv
source .venv/bin/activate
```
You need python version 3.11 and detectron2 with the DensePose project installed. These packages are sensitive to mutual version mismatch. At the same time it might be useful to see how the packages are build and configured and so we clone the repository first in the _external_ directory and install it safely in editable mode. You might also want to include the _--no-deps_ flag in order to refrain from pip playing with your effortfully chosen dependencies. All the prerequisits should be loaded at this point anyway.

```
cd external
git clone https://github.com/facebookresearch/detectron2.git
pip install --no-build-isolation -e detectron2
cd detectron2/projects/DensePose
pip install --no-build-isolation -e detectron2/projects/DensePose
```

### On your local machine
In the case you want to run the repo on your personal machine or a different server, here is the general setup manual.

#### Compute

All the tasks are computation intensive and it is recommended to use a solid GPU and RAM.

#### Environment

You are going to need to have many requirements. You can freshly install these into you virtual environment with python 3.11 made in the project's root. Here is how you create one and activate it. You can also use ```conda``` as a package manager (Which might be the winner here as you need to install a lot of packages with complex dependencies).

```
python3.11 -m venv .venv 
source .venv/bin/activate
```

First, you need to get the right PyTorch version. You need to find your CUDA version through 

```
nvidia-smi
```

You should look at https://download.pytorch.org/whl/ to see the newest version that is still smaller than your CUDA version. 

and choose the correct version of PyTorch 2.4.0 so that it communicates well with our GPU (PyTorch version <= GPU version of CUDA). Here is an example for the version 12.4.0 CUDA on our GPU that uses the PyTorch made for the 12.1.0 CUDA as that is the newest stable version of it. You should replace the _cu121_ with whatevetr the version you need is. 
```
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```
Note that with conda you would use:
```
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
The detectron2 project and especially DensePose libraries provide a tight space to manipulate and that is why it is best to install the full list of libraries with the right verisions straight away. To do that easily you can use the prepared list of requirements.

For the best dependancy handling, you should install PyTorch and all the requirements in one command using:

```
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

Or

```
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia --file requirements.txt
```

More at https://pytorch.org/get-started/previous-versions/
and https://anaconda.org/pytorch/pytorch



You also need and detectron2 with the DensePose project installed. These packages are sensitive to mutual version mismatch. At the same time it might be useful to see how the packages are build and configured and so we clone the repository first in the _external_ directory and install it safely in editable mode. You might also want to include the ```--no-deps``` flag in order to refrain from pip playing with your effortfully chosen dependencies. All the prerequisits should be loaded at this point anyway.

```
cd external
git clone https://github.com/facebookresearch/detectron2.git
pip install --no-build-isolation --no-deps -e detectron2
cd detectron2/projects/DensePose
pip install --no-build-isolation --no-deps -e detectron2/projects/DensePose
```

You should be ready to go!


## Installing data
### COCO Dataset
We choosed coco dataset for our research because it is a staple amongst research while providing tricky scenarios where we can evaluate the performance of our methods compared to the models such as DensePose fo NLF. We choose specifivaly the COCO 2014 minival evaluation dataset. It is crucial that the models we are trying to improve have not been trained on our dataset as we would be fiting for a better answer with someone who has seen the answers otherwise. 

You can read more about the coco dataset in their [paper](https://arxiv.org/pdf/1405.0312).

#### Pictures

You can download the dataset on http://images.cocodataset.org/zips/val2014.zip. Please unzip it and move it to the ```data``` foder in the project or use a symlink as you are going to need access to those pictures. 

We use just a subset of the images that have been chosen to be the most accurately annotated. That is why we share the 

#### DensePose annotations

To train the DensePose model, facebook annotated the COCO 2014 datasen with a UV map that connects every pixel in the image with a corresponding point on a canonical human body of SMPL model. 

They have shared all the annotations and dataset information here https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_DATASETS.md. You can also use this command:

```
cd data
curl -L -o densepose_minival2014_cse_online.json "https://dl.fbaipublicfiles.com/densepose/annotations/coco_cse/densepose_minival2014_cse.json"
```
You should save this into ```data``` directory. 

### DensePose model weights
The model weights are on https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_CSE.md. We opted for _R_101_FPN_DL_soft_s1x_ as that is the biggest model with the best performance. You can install the model weights at we use by: 
```
cd models/densepose
curl -L -o model_final_1d3314.pkl "https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x/250713061/model_final_1d3314.pkl"
```
They should come in ```models/densepose```.

### SMPL and SMPLX

The SMPL model is presented here https://smpl.is.tue.mpg.de/index.html. To download any models form this company you need to register first. You can also download it by
```
cd models
curl -L -o SMPL_python_v.1.1.0.zip "https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip"
```
You should put the ```smpl``` folder that is inside this wrapper directly in the ```models``` folder in the project. 

You might also want the SMPLX models for expresiveness. In that case take a look at https://smpl-x.is.tue.mpg.de/index.html or use the command
```
cd models
curl -L -o SMPL_python_v.1.1.0.zip "https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip"`
```
The folder called ```smplx``` from inside the downloaded folder belongs to the ```models``` in our project. 



## Credits and licenses
This project utilizes the following open-source libraries:
* [Detectron2](https://github.com/facebookresearch/detectron2) (Apache 2.0)
* [DensePose](https://github.com/facebookresearch/densepose) (Apache 2.0)
* [COCO](https://github.com/cocodataset/cocoapi) (Apache 2.0)