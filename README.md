# Reproduction_of_MCAN
This is the reproduction of MCAN from paper in ACL 2021: "Multimodal Fusion with Co-Attention Networks for Fake News Detection".

本项目是结合weibo和twitter两个数据集对官方MCAN代码进行的注释和复现。
论文作者给出的原始代码地址[https://github.com/wuyang45/MCAN_code]



# create the env
首先创建代码所需的环境，具体所需packages见官方代码地址。

# process dataset
首先从下面的地址下载所需数据集：
weibo[https://pan.baidu.com/s/1S0OxCWRvXsP2cOWdDt_BRg],提取码：4j7p
twitter[https://pan.baidu.com/s/1GOLqfw4n0XaR33AR7fSqVg]，提取码：fww9

然后运行下面的代码处理数据集，得到满足MCAN输入格式的文件：
```
python text_process.py
```


# References
If you are insterested in this work, and want to use the codes or results in this repository, please star this repository and cite by:
```

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
