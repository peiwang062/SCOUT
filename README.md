# SCOUT

![cf](/Figs/cf.png)

This repository contains the source code accompanying our CVPR 2020 paper.

**[SCOUT: Self-Aware Discriminant Counterfactual Explanations](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_SCOUT_Self-Aware_Discriminant_Counterfactual_Explanations_CVPR_2020_paper.html)**  
[Pei Wang](http://www.svcl.ucsd.edu/~peiwang), [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno).  
In Computer Vision and Pattern Recognition, 2020.

```
@InProceedings{wang2020scout,
author = {Wang, Pei and Vasconcelos, Nuno},
title = {SCOUT: Self-Aware Discriminant Counterfactual Explanations},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Requirements

1. The project was implemented and tested in Python 3.5 and Pytorch 0.4. The higher versions should work after minor modification.
2. Other common modules like numpy, pandas and seaborn for visualization.
3. NVIDIA GPU and cuDNN are required to have fast speeds. For now, CUDA 8.0 with cuDNN 6.0.20 has been tested. The other versions should be working.


## Datasets

[CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [ADE20K](http://sceneparsing.csail.mit.edu/) are used. Please organize them as below after download,


```
cub200
|_ CUB_200_2011
  |_ attributes
  |_ images
  |_ parts
  |_ train_test_split.txt
  |_ ...
```

```
ade
|_ ADEChallengeData2016
  |_ annotations
  |_ images
  |_ objectInfo 150.txt
  |_ sceneCategories.txt
```

## Implementation details

### data preparation

build train/validation/test sets,

```
make_cub_list.py
make_ade_list.py
```

compute ground truth parts on CUB200 and objects on ADE20K,

```
make_gt_cub.py
make_gt_ade.py
```

prepare attribute location data on CUB200

```
get_gt_partLocs.py
```

### training

Two types of models need to be trained, the standard CNN classifier and [Hardness predictor](http://openaccess.thecvf.com/content_ECCV_2018/html/Pei_Wang_Towards_Realistic_Predictors_ECCV_2018_paper.html). We separately wrote the code for each experiment. For the classifier,
```
./cub200/train_cub_alexnet.py
./cub200/train_cub_vgg.py
./cub200/train_cub_res.py
./ade/train_ade_alexnet.py
./ade/train_ade_vgg.py
./ade/train_ade_res.py
```
for the hardness predictor,
```
./cub200/train_hp_cub_vgg.py
./ade/train_hp_ade_vgg.py
```

### Inference

1. To reproduce results on sec 5.1
```
python cf_ss_vgg_cub.py --student=beginners --maps=a
python cf_ss_vgg_cub.py --student=beginners --maps=ab
python cf_ss_vgg_cub.py --student=beginners --maps=abs
python cf_cs_vgg_cub.py --student=beginners --maps=abs
python cf_es_vgg_cub.py --student=beginners --maps=abs
python cf_ss_vgg_cub.py --student=advanced --maps=a
python cf_ss_vgg_cub.py --student=advanced --maps=ab
python cf_ss_vgg_cub.py --student=advanced --maps=abs
python cf_cs_vgg_cub.py --student=advanced --maps=abs
python cf_es_vgg_cub.py --student=advanced --maps=abs
```

2. To reproduce results on sec 5.2
```
python cf_ss_vgg_cub.py --student=beginners
python cf_ss_vgg_cub.py --student=advanced
python cf_ss_res_cub.py --student=beginners
python cf_ss_res_cub.py --student=advanced
python cf_PIoU_ss_vgg_cub.py --student=beginners
python cf_PIoU_ss_vgg_cub.py --student=advanced
```

### pretrained models

The [pre-trained models](https://drive.google.com/drive/folders/1fh1HMqjrFFctkTgjYvylEiUhEQP3aZOg?usp=sharing) for all experiments are availiable.


## Disclaimer

The code has not been purified and polished so far. We will do this soon. For questions, feel free to reach out
```
Pei Wang: peiwang062@gmail.com
```


## References

[1] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra.  Grad-cam:  Visual explanations from deep networks via gradient-based localization.  In Proceedings of the IEEE International Conference on Computer Vision, pages 618–626, 2017.

[2] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 3319–3328. JMLR. org,4662017.

[3] Pei Wang and Nuno Vasconcelos. Towards realistic predictors. In The European Conference on Computer Vision, 2018.

[4] Welinder P., Branson S., Mita T., Wah C., Schroff F., Belongie S., Perona, P. “Caltech-UCSD Birds 200”. California Institute of Technology. CNS-TR-2010-001. 2010.

[5] Bolei  Zhou,  Hang  Zhao,  Xavier  Puig,  Sanja  Fidler,  Adela  Barriuso,  and  Antonio  Torralba.   Scene parsing through ade20k
