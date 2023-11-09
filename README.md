# Reproduction_of_MCAN
This is the reproduction of MCAN from paper in ACL 2021: "Multimodal Fusion with Co-Attention Networks for Fake News Detection".

本项目是结合weibo和twitter两个数据集对官方MCAN代码进行的注释和复现。
论文作者给出的原始代码地址[官方代码](https://github.com/wuyang45/MCAN_code)



# create the env
首先创建代码所需的环境，具体所需packages见官方代码地址。

# process dataset
首先从下面的地址下载所需数据集：
weibo数据集[下载地址](https://pan.baidu.com/s/1S0OxCWRvXsP2cOWdDt_BRg),提取码：4j7p
twitter数据集[下载地址](https://pan.baidu.com/s/1GOLqfw4n0XaR33AR7fSqVg)，提取码：fww9

然后运行下面的代码处理数据集，得到满足MCAN输入格式的文件：
```
python text_process.py
```


# References
If you are insterested in this work, and want to use the dataset or codes in this repository, please star this repository and cite by:
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
