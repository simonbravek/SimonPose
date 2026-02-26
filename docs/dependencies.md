# Dependencies
## Environment essentials
These are all the working releases that you could try out as well if you feel but we recommend sticking with what we use above. 

```
ml PyTorch/2.2.1-foss-2023b-CUDA-12.4.0 torchvision/0.17.1-foss-2023b-CUDA-12.4.0 OpenCV/4.10.0-foss-2023b-CUDA-12.4.0-contrib Albumentations/1.4.4-foss-2023b-CUDA-12.4.0 pycocotools/2.0.7-foss-2023b matplotlib/3.8.2-gfbf-2023b

ml PyTorch/2.3.0-foss-2023b torchvision/0.18.0-foss-2023b-CUDA-12.4.0 OpenCV/4.10.0-foss-2023b-CUDA-12.4.0-contrib Albumentations/1.4.4-foss-2023b-CUDA-12.4.0 pycocotools/2.0.7-foss-2023b matplotlib/3.8.2-gfbf-2023b

ml PyTorch/2.4.0-foss-2023b-CUDA-12.4.0 torchvision/0.19.0-foss-2023b-CUDA-12.4.0 OpenCV/4.10.0-foss-2023b-CUDA-12.4.0-contrib Albumentations/1.4.4-foss-2023b-CUDA-12.4.0 pycocotools/2.0.7-foss-2023b matplotlib/3.8.2-gfbf-2023b

ml PyTorch/2.5.1-foss-2023b-CUDA-12.4.0 torchvision/0.20.1-foss-2023b-CUDA-12.4.0 OpenCV/4.10.0-foss-2023b-CUDA-12.4.0-contrib Albumentations/1.4.4-foss-2023b-CUDA-12.4.0 pycocotools/2.0.7-foss-2023b matplotlib/3.8.2-gfbf-2023b
```

## DensePose dataset alternative
For purposes where the fractional body parts approach is desired we can use the original verison that did not use: https://github.com/facebookresearch/DensePose/tree/main/DensePoseData. Please note that this is the first version of _detectron_ and so it should not be used for anything but the dataset as all the projects in there are outdated and not supported. The annotations for this verison can be downloaded on https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_minival.json.
