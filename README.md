# Reproduction_of_MCAN
This is the reproduction of MCAN from paper in ACL 2021: "Multimodal Fusion with Co-Attention Networks for Fake News Detection".

本项目是结合weibo和twitter两个数据集对官方MCAN代码进行的注释和复现。
论文作者给出的原始代码地址[官方代码](https://github.com/wuyang45/MCAN_code)


# Create the env
首先创建代码所需的环境，推荐使用anaconda创建虚拟环境。代码运行所需要的关键包已在requirements.txt中列出。

The python version is python-3.8.16. The detailed version of some packages is available in requirements.txt. You can install all the required packages using the following command:
```
conda install --yes --file requirements.txt
```

# Process dataset

数据集可以从这个项目中获取：[weibo和twitter数据集](https://github.com/plw-study/MRML)

运行下面的代码处理原始数据，得到满足MCAN输入格式的文件：
```
python text_process.py
```
上述代码运行完成之后，会在当前目录下的processd_data中生成每个数据集的train.txt和test.txt文件。

# Download pre_trained models
运行代码需要预训练好的 bert-base-chinese, bert-base-multilingual-cased 以及 pytorch的vgg19 模型。

下载之后将模型放入当前目录下的models文件夹中。

# Run
直接运行py文件即可开始训练：
```
python MCAN_reproduction.py
```

# References
If you are insterested in this work, and want to use the dataset or codes in this repository, please star this repository and cite by:

```
@article{PENG2024103564,
title = {Not all fake news is semantically similar: Contextual semantic representation learning for multimodal fake news detection},
journal = {Information Processing & Management},
volume = {61},
number = {1},
pages = {103564},
year = {2024},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2023.103564},
url = {https://www.sciencedirect.com/science/article/pii/S0306457323003011},
author = {Liwen Peng and Songlei Jian and Zhigang Kan and Linbo Qiao and Dongsheng Li},
keywords = {Fake news detection, Multimodal learning, Social network, Representation learning, Deep learning}
}
```

```
@INPROCEEDINGS{peng-MRML,
  author={Peng, Liwen and Jian, Songlei and Li, Dongsheng and Shen, Siqi},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={MRML: Multimodal Rumor Detection by Deep Metric Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096188}
}
```

```
@inproceedings{wu2021multimodal,
        title={Multimodal fusion with co-attention networks for fake news detection},
        author={Wu, Yang and Zhan, Pengwei and Zhang, Yunjian and Wang, Liming and Xu, Zhen},
        booktitle={Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
        pages={2560--2569},
        year={2021}
}
```
