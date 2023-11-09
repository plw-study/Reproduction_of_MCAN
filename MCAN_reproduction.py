# -*- coding: utf-8 -*-#
from datetime import date, datetime
import os
import random
import math

from adabelief_pytorch import AdaBelief
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ray import tune
# from ray.tune.schedulers import AsyncHyperBandScheduler
# from ray.tune.suggest import ConcurrencyLimiter
# from ray.tune.suggest import Repeater
# from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune import Callback
# from ray.tune import JupyterNotebookReporter

from scipy.fftpack import fft, dct
import seaborn as sns
from sklearn.metrics import classification_report
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# from torch.cuda.amp import autocast, GradScaler
from torch.optim import *
from torch.utils.data import RandomSampler, SequentialSampler, Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR
import torchvision
from tqdm.notebook import tqdm, trange
from transformers import BertTokenizer, BertConfig, BertModel, get_linear_schedule_with_warmup
from tabulate import tabulate

# from IPython.display import display, HTML

pd.options.display.max_columns = None

random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# reporter = JupyterNotebookReporter(overwrite=False, max_progress_rows=1000)


def process_dct_img(img):
    img = img.numpy()  # size = [1, 224, 224]
    height = img.shape[1]
    width = img.shape[2]
    # print('height:{}'.format(height))
    N = 8
    step = int(height / N)  # 28

    dct_img = np.zeros((1, N * N, step * step, 1), dtype=np.float32)  # [1,64,784,1]
    fft_img = np.zeros((1, N * N, step * step, 1))
    # print('dct_img:{}'.format(dct_img.shape))

    i = 0
    for row in np.arange(0, height, step):
        for col in np.arange(0, width, step):
            block = np.array(img[:, row:(row + step), col:(col + step)], dtype=np.float32)
            # print('block:{}'.format(block.shape))
            block1 = block.reshape(-1, step * step, 1)  # [batch_size,784,1]
            dct_img[:, i, :, :] = dct(block1)  # [batch_size, 64, 784, 1]

            i += 1

    # for i in range(64):
    fft_img[:, :, :, :] = fft(dct_img[:, :, :, :]).real  # [batch_size,64, 784,1]

    fft_img = torch.from_numpy(fft_img).float()  # [batch_size, 64, 784, 1]
    new_img = F.interpolate(fft_img, size=[250, 1])  # [batch_size, 64, 250, 1]
    new_img = new_img.squeeze(0).squeeze(-1)  # torch.size = [64, 250]

    return new_img


class MyDataset(Dataset):
    def __init__(self, data, VOCAB, max_sen_len, transform_vgg=None, transform_dct=None):
        super(MyDataset, self).__init__()

        self.transform_vgg = transform_vgg
        self.transform_dct = transform_dct
        self.tokenizer = BertTokenizer.from_pretrained(VOCAB)
        self.max_sen_len = max_sen_len

        self.post_id = torch.from_numpy(data['post_id'])
        self.tweet_content = data['post_content']
        # self.image = list(self.transform(data['image']))
        self.image = list(data['image'])
        self.label = torch.from_numpy(data['label'])  # type:int

    def __getitem__(self, idx):
        content = str(self.tweet_content[idx])
        text_content = self.tokenizer.encode_plus(content, add_special_tokens=True, padding='max_length',
                                                  truncation=True, max_length=self.max_sen_len, return_tensors='pt')

        dct_img = self.transform_dct(self.image[idx].convert('L'))
        dct_img = process_dct_img(dct_img)

        return {
            "text_input_ids": text_content["input_ids"].flatten().clone().detach().type(torch.LongTensor),
            "attention_mask": text_content["attention_mask"].flatten().clone().detach().type(torch.LongTensor),
            "token_type_ids": text_content["token_type_ids"].flatten().clone().detach().type(torch.LongTensor),
            "image": self.transform_vgg(self.image[idx]),
            "dct_img": dct_img,
            "post_id": self.post_id[idx],
            "label": self.label[idx],
        }

    def __len__(self):
        return len(self.label)


class vgg(nn.Module):
    """
    obtain visual feature
    """

    def __init__(self, model_dim, pthfile):
        super(vgg, self).__init__()
        self.model_dim = model_dim
        self.pthfile = pthfile

        # image
        vgg_19 = torchvision.models.vgg19(pretrained=False)
        vgg_19.load_state_dict(torch.load(self.pthfile))

        self.feature = vgg_19.features
        self.classifier = nn.Sequential(*list(vgg_19.classifier.children())[:-3])
        pretrained_dict = vgg_19.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # delect the last layer
        model_dict.update(pretrained_dict)  # update
        self.classifier.load_state_dict(model_dict)  # load the new parameter

    def forward(self, img):
        # image
        # image = self.vgg(img) #[batch, num_ftrs]
        img = self.feature(img)
        img = img.view(img.size(0), -1)
        image = self.classifier(img)

        return image


class multimodal_attention(nn.Module):
    """
    dot-product attention mechanism
    """

    def __init__(self, attention_dropout=0.5):
        super(multimodal_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.matmul(q, k.transpose(-2, -1))
        # print('attention.shape:{}'.format(attention.shape))
        if scale:
            attention = attention * scale

        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        # print('attention.shftmax:{}'.format(attention))
        attention = self.dropout(attention)
        attention = torch.matmul(attention, v)
        # print('attn_final.shape:{}'.format(attention.shape))

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_heads=8, dropout=0.5):
        super(MultiHeadAttention, self).__init__()

        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(1, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = multimodal_attention(dropout)
        self.linear_final = nn.Linear(model_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query
        query = query.unsqueeze(-1)
        key = key.unsqueeze(-1)
        value = value.unsqueeze(-1)
        # print("query.shape:{}".format(query.shape))

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        # batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        # print('key.shape:{}'.format(key.shape))

        # split by heads
        key = key.view(-1, num_heads, self.model_dim, dim_per_head)
        value = value.view(-1, num_heads, self.model_dim, dim_per_head)
        query = query.view(-1, num_heads, self.model_dim, dim_per_head)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        attention = self.dot_product_attention(query, key, value,
                                               scale, attn_mask)

        attention = attention.view(-1, self.model_dim, dim_per_head * num_heads)
        # print('attention_con_shape:{}'.format(attention.shape))

        # final linear projection
        output = self.linear_final(attention).squeeze(-1)
        # print('output.shape:{}'.format(output.shape))
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output


class PositionalWiseFeedForward(nn.Module):
    """
    Fully-connected network
    """

    def __init__(self, model_dim=256, ffn_dim=2048, dropout=0.5):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x

        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        output = x
        return output


class multimodal_fusion_layer(nn.Module):
    """
    A layer of fusing features
    """

    def __init__(self, model_dim=256, num_heads=8, ffn_dim=2048, dropout=0.5):
        super(multimodal_fusion_layer, self).__init__()
        self.attention_1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attention_2 = MultiHeadAttention(model_dim, num_heads, dropout)

        self.feed_forward_1 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.feed_forward_2 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

        self.fusion_linear = nn.Linear(model_dim * 2, model_dim)

    def forward(self, image_output, text_output, attn_mask=None):
        output_1 = self.attention_1(image_output, text_output, text_output,
                                    attn_mask)

        output_2 = self.attention_2(text_output, image_output, image_output,
                                    attn_mask)

        # print('attention out_shape:{}'.format(output.shape))
        output_1 = self.feed_forward_1(output_1)
        output_2 = self.feed_forward_2(output_2)

        output = torch.cat([output_1, output_2], dim=1)
        output = self.fusion_linear(output)

        return output


def ConvBNRelu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )


def ConvBNRelu2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DctStem(nn.Module):
    def __init__(self, kernel_sizes, num_channels):
        super(DctStem, self).__init__()
        self.convs = nn.Sequential(
            ConvBNRelu2d(in_channels=1,
                         out_channels=num_channels[0],
                         kernel_size=kernel_sizes[0]),
            ConvBNRelu2d(
                in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=kernel_sizes[1],
            ),
            ConvBNRelu2d(
                in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=kernel_sizes[2],
            ),
            nn.MaxPool2d((1, 2)),
        )

    def forward(self, dct_img):
        x = dct_img.unsqueeze(1)
        img = self.convs(x)
        img = img.permute(0, 2, 1, 3)

        return img


class DctInceptionBlock(nn.Module):
    def __init__(
            self,
            in_channel=128,
            branch1_channels=[64],
            branch2_channels=[48, 64],
            branch3_channels=[64, 96, 96],
            branch4_channels=[32],
    ):
        super(DctInceptionBlock, self).__init__()

        self.branch1 = ConvBNRelu2d(in_channels=in_channel,
                                    out_channels=branch1_channels[0],
                                    kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNRelu2d(in_channels=in_channel,
                         out_channels=branch2_channels[0],
                         kernel_size=1),
            ConvBNRelu2d(
                in_channels=branch2_channels[0],
                out_channels=branch2_channels[1],
                kernel_size=3,
                padding=(0, 1),
            ),
        )

        self.branch3 = nn.Sequential(
            ConvBNRelu2d(in_channels=in_channel,
                         out_channels=branch3_channels[0],
                         kernel_size=1),
            ConvBNRelu2d(
                in_channels=branch3_channels[0],
                out_channels=branch3_channels[1],
                kernel_size=3,
                padding=(0, 1),
            ),
            ConvBNRelu2d(
                in_channels=branch3_channels[1],
                out_channels=branch3_channels[2],
                kernel_size=3,
                padding=(0, 1),
            ),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1)),
            ConvBNRelu2d(in_channels=in_channel,
                         out_channels=branch4_channels[0],
                         kernel_size=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        # y = x
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = out.permute(0, 2, 1, 3)

        return out


class DctCNN(nn.Module):
    def __init__(self,
                 model_dim,
                 dropout,
                 kernel_sizes,
                 num_channels,
                 in_channel=128,
                 branch1_channels=[64],
                 branch2_channels=[48, 64],
                 branch3_channels=[64, 96, 96],
                 branch4_channels=[32],
                 out_channels=64):
        super(DctCNN, self).__init__()

        self.stem = DctStem(kernel_sizes, num_channels)

        self.InceptionBlock = DctInceptionBlock(
            in_channel,
            branch1_channels,
            branch2_channels,
            branch3_channels,
            branch4_channels,
        )

        self.maxPool = nn.MaxPool2d((1, 122))

        self.dropout = nn.Dropout(dropout)

        self.conv = ConvBNRelu2d(branch1_channels[-1] + branch2_channels[-1] +
                                 branch3_channels[-1] + branch4_channels[-1],
                                 out_channels,
                                 kernel_size=1)

    def forward(self, dct_img):
        dct_f = self.stem(dct_img)
        x = self.InceptionBlock(dct_f)
        x = self.maxPool(x)
        x = x.permute(0, 2, 1, 3)
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3)
        x = x.squeeze(-1)

        x = x.reshape(-1, 4096)

        return x


class NetShareFusion(nn.Module):
    def __init__(self,
                 CASED,
                 pthfile,
                 kernel_sizes,
                 num_channels,
                 model_dim,
                 drop_and_BN,
                 bert_dim=768,
                 img_size=250,
                 num_labels=2,
                 num_layers=1,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.5):

        super(NetShareFusion, self).__init__()

        self.CASED = CASED
        self.model_dim = model_dim
        self.pthfile = pthfile
        self.drop_and_BN = drop_and_BN

        # text
        self.config = BertConfig.from_pretrained(self.CASED)

        self.bert = BertModel.from_pretrained(self.CASED, config=self.config)
        self.linear_text = nn.Linear(bert_dim, model_dim)
        self.bn_text = nn.BatchNorm1d(model_dim)

        self.dropout = nn.Dropout(dropout)

        # image
        self.vgg = vgg(model_dim, pthfile)
        self.linear_image = nn.Linear(4096, model_dim)
        self.bn_vgg = nn.BatchNorm1d(model_dim)

        # dct_image
        self.dct_img = DctCNN(model_dim,
                              dropout,
                              kernel_sizes,
                              num_channels,
                              in_channel=128,
                              branch1_channels=[64],
                              branch2_channels=[48, 64],
                              branch3_channels=[64, 96, 96],
                              branch4_channels=[32],
                              out_channels=64)
        self.linear_dct = nn.Linear(4096, model_dim)
        self.bn_dct = nn.BatchNorm1d(model_dim)

        # multimodal fusion
        self.fusion_layers = nn.ModuleList([
            multimodal_fusion_layer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # classifier
        self.linear1 = nn.Linear(model_dim, 35)
        self.bn_1 = nn.BatchNorm1d(35)
        self.linear2 = nn.Linear(35, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def drop_BN_layer(self, x, part='dct'):
        if part == 'dct':
            bn = self.bn_dct
        elif part == 'vgg':
            bn = self.bn_vgg
        elif part == 'bert':
            bn = self.bn_text

        if self.drop_and_BN == 'drop-BN':
            x = self.dropout(x)
            x = bn(x)
        elif self.drop_and_BN == 'BN-drop':
            x = bn(x)
            x = self.dropout(x)
        elif self.drop_and_BN == 'drop-only':
            x = self.dropout(x)
        elif self.drop_and_BN == 'BN-only':
            x = bn(x)
        elif self.drop_and_BN == 'none':
            pass

        return x

    def forward(self, text_input_ids, token_type_ids, attention_mask, image,
                dct_img, attn_mask):

        # textual feature
        bert_output = self.bert(input_ids=text_input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        text_output = bert_output[1]  # the representation of the whole sentence
        # print('bert_output:{}, shape:{}'.format(text_output, text_output.shape))
        text_output = F.relu(self.linear_text(text_output))
        text_output = self.drop_BN_layer(text_output, part='bert')
        #         print('text_output:{}'text_output.shape)

        # visual feature
        output = self.vgg(image)
        output = F.relu(self.linear_image(output))
        output = self.drop_BN_layer(output, part='vgg')

        # dct_feature
        dct_out = self.dct_img(dct_img)
        dct_out = F.relu(self.linear_dct(dct_out))
        dct_out = self.drop_BN_layer(dct_out, part='dct')

        for fusion_layer in self.fusion_layers:
            output = fusion_layer(output, dct_out, attn_mask)

        for fusion_layer in self.fusion_layers:
            output = fusion_layer(output, text_output, attn_mask)
            # print('fusion output_shape:{}'.format(output.shape))

        output = F.relu(self.linear1(output))
        output = self.dropout(output)
        # output = self.bn_1(output)
        output = self.linear2(output)
        # print('output_size:{}'.format(output.shape))
        y_pred_prob = self.softmax(output)

        return output, y_pred_prob


class EarlyStopping:
    """Early stops the training if test acc doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time test acc improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.test_acc_max = 0

        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, test_acc, test_recall_values):

        score = test_acc

        if self.best_score is None:
            self.best_score = score
            self.update_max_test_acc(test_acc)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}. (Best: {self.test_acc_max:.6f})"
                )
            if self.counter >= self.patience:
                self.trace_func(
                    f"**EarlyStopping Triggered: test accuracy stuck at {self.test_acc_max:.6f} for {self.patience} epoch(es)."
                )
                self.early_stop = True
        else:
            self.best_score = score
            self.update_max_test_acc(test_acc)
            self.counter = 0

    def update_max_test_acc(self, test_acc):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Test accuracy increased ({self.test_acc_max:.6f} --> {test_acc:.6f})."
            )
        self.test_acc_max = test_acc


class TrainALL(tune.Trainable):
    def get_dataloader(self):

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        if self.dataset_name == 'weibo':
            import data_process_weibo as pro
        elif self.dataset_name == 'twitter':
            import data_process_twitter as pro
        else:
            raise ValueError('ERROR! dataset_name must be weibo or twitter!')

        image_list = pro.read_images(pro.image_file_list)

        train_data, train_data_num = pro.get_data('train', image_list)
        test_data, valid_data_num = pro.get_data('test', image_list)

        if self.dataset_name == 'twitter':
            transform_vgg = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.454, 0.440, 0.423], [0.282, 0.278, 0.278])
            ])
            transform_dct = transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.ToTensor()
                 ])

        elif self.dataset_name == 'weibo':
            transform_vgg = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            transform_dct = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            raise 'Dataset Error'

        train_dataset = MyDataset(data=train_data,
                                  VOCAB=self.VOCAB,
                                  max_sen_len=self.max_sen_len,
                                  transform_vgg=transform_vgg,
                                  transform_dct=transform_dct)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler,
                                  batch_size=self.train_bs,
                                  num_workers=1,
                                  drop_last=True)  # twitter-bn处理时候报错，说是batch有单个数据，所以扔掉last batch

        test_dataset = MyDataset(data=test_data,
                                 VOCAB=self.VOCAB,
                                 max_sen_len=self.max_sen_len,
                                 transform_vgg=transform_vgg,
                                 transform_dct=transform_dct)
        #         test_sampler = SequentialSampler(test_dataset)
        test_sampler = RandomSampler(test_dataset)
        test_loader = DataLoader(dataset=test_dataset,
                                 sampler=test_sampler,
                                 batch_size=self.test_bs,
                                 num_workers=1)  # test时候不能扔掉last batch
        return train_loader, test_loader

    def get_optimizer(self):
        no_decay = [
            "bias",
            "gamma",
            "beta",
            "LayerNorm.weight",
            "bn_text.weight",
            "bn_dct.weight",
            "bn_1.weight",
        ]

        bert_param_optimizer = list(self.model.bert.named_parameters())
        vgg_param_optimizer = list(self.model.vgg.named_parameters())
        dtcconv_param_optimizer = list(self.model.dct_img.named_parameters())
        fusion_param_optimizer = list(
            self.model.fusion_layers.named_parameters()
        )
        linear_param_optimizer = (
                list(self.model.linear_text.named_parameters())
                + list(self.model.linear_image.named_parameters())
                + list(self.model.linear_dct.named_parameters())
        )
        classifier_param_optimizer = list(self.model.linear1.named_parameters()) + list(
            self.model.linear2.named_parameters()
        )
        optimizer_grouped_parameters = [
            # bert_param_optimizer
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay,
             "lr": self.bert_learning_rate, },
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": self.bert_learning_rate, },
            # vgg_param_optimizer
            {"params": [p for n, p in vgg_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay,
             "lr": self.vgg_learning_rate, },
            {"params": [p for n, p in vgg_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": self.vgg_learning_rate, },
            # dtcconv_param_optimizer
            {"params": [p for n, p in dtcconv_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay,
             "lr": self.dtcconv_learning_rate, },
            {"params": [p for n, p in dtcconv_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": self.dtcconv_learning_rate, },
            # fusion_param_optimizer
            {"params": [p for n, p in fusion_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay,
             "lr": self.fusion_learning_rate, },
            {"params": [p for n, p in fusion_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": self.fusion_learning_rate, },
            # linear_param_optimizer
            {"params": [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay,
             "lr": self.linear_learning_rate, },
            {"params": [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": self.linear_learning_rate, },
            # classifier_param_optimizer
            {"params": [p for n, p in classifier_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay,
             "lr": self.classifier_learning_rate, },
            {"params": [p for n, p in classifier_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": self.classifier_learning_rate, },
        ]

        if self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                #                 filter(lambda p: filter(lambda x: x['params'].requires_grad, p), optimizer_grouped_parameters),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                #                 filter(lambda p: filter(lambda x: x['params'].requires_grad, p), optimizer_grouped_parameters),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                #                 filter(lambda p: filter(lambda x: x['params'].requires_grad, p), optimizer_grouped_parameters),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "AdaBelief":
            from adabelief_pytorch import AdaBelief
            optimizer = AdaBelief(
                optimizer_grouped_parameters,
                #                 filter(lambda p: filter(lambda x: x['params'].requires_grad, p), optimizer_grouped_parameters),
                lr=self.learning_rate,
                eps=1e-10,  # or 1e-16
                betas=(0.9, 0.999),
                weight_decouple=True,
                rectify=False)
        else:
            raise 'optimizer WRONG'
        return optimizer

    def get_scheduler(self):
        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(self.train_loader) * self.epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=round(total_steps * self.warm_up_percentage),
            num_training_steps=total_steps
        )
        return scheduler

    def init_network(self, exclude_list=['bert', 'vgg']):
        if self.init_method != 'default':
            for name, w in self.model.named_parameters():
                cross = [val for val in exclude_list if val in name.split('.')]
                if cross == []:  # 对于embedding，保留预训练的embedding
                    if [val for val in ['bn_text', 'bn_vgg', 'bn_dct', 'bn_1', 'layer_norm'] if
                        val in name.split('.')] == []:
                        if 'weight' in name:

                            if self.init_method == 'xavier-normal':
                                nn.init.xavier_normal_(w)
                            elif self.init_method == 'xavier-uniform':
                                nn.init.xavier_uniform_(w)
                            elif self.init_method == 'kaiming-normal':
                                nn.init.kaiming_normal_(w)
                            elif self.init_method == 'kaiming-uniform':
                                nn.init.kaiming_uniform_(w)
                            else:
                                pass
                        elif 'bias' in name:
                            nn.init.constant_(w, 0)
                        else:
                            pass

    def get_model(self):
        model = NetShareFusion(CASED=self.CASED,
                               pthfile=self.pthfile,
                               kernel_sizes=self.kernel_sizes,
                               num_channels=self.num_channels,
                               num_layers=self.num_layers,
                               num_heads=self.num_heads,
                               model_dim=self.model_dim,
                               dropout=self.dropout,
                               drop_and_BN=self.drop_and_BN)

        if self.FREEZE_BERT:
            for name, param in model.named_parameters():
                if "bert" in name:
                    param.requires_grad = False

        if self.FREEZE_VGG:
            for name, param in model.named_parameters():
                if "vgg" in name:
                    param.requires_grad = False

        return model

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1)
        labels_flat = labels
        return np.sum(pred_flat == labels_flat) / len(labels)

    def config_check(self):
        if self.dataset_name == 'weibo' and 'multilingual' in self.CASED:
            raise ('Using weibo dataset with multilingual model!')
        if self.dataset_name == 'twitter' and 'chinese' in self.CASED:
            raise ('Using twitter dataset with chinese model!')

    def setup(self, config):
        self.config = config

        # 这里要把路径替换成本机的bert-base和vgg模型的绝对路径
        # 同理，还有vgg19的模型

        # 使用哪一个model需要配合要处理的数据集选择，
        # weibo数据集用，bert-base-chinese
        # twitter数据集用 bert-base-multilingual-cased

        # self.CASED = '/home/xxxx/models/bert-base-chinese/'  # multilingual-cased
        # self.VOCAB = '/home/xxxx/models/bert-base-chinese/vocab.txt'
        self.CASED = '/home/xxxx/models/bert-base-multilingual-cased/'  # multilingual-cased
        self.VOCAB = '/home/xxxx/models/bert-base-multilingual-cased/vocab.txt'
        self.pthfile = '/home/xxxx/models/vgg19-dcbb9e9d.pth'
        self.save_root = './results/'

        self.init_method = config.get("init_method")
        self.max_grad_norm = 1.0
        self.warm_up_percentage = 0.1
        self.early_stopping_patience = config.get("early_stopping_patience")
        self.early_stopping = EarlyStopping(patience=self.early_stopping_patience, verbose=True)

        self.bert_learning_rate = config.get("bert_learning_rate")
        self.vgg_learning_rate = config.get("vgg_learning_rate")
        self.dtcconv_learning_rate = config.get("dtcconv_learning_rate")
        self.fusion_learning_rate = config.get("fusion_learning_rate")
        self.linear_learning_rate = config.get("linear_learning_rate")
        self.classifier_learning_rate = config.get("classifier_learning_rate")

        self.FREEZE_BERT = config.get("FREEZE_BERT")
        self.FREEZE_VGG = config.get("FREEZE_VGG")

        self.seed = config.get("seed")
        self.kernel_sizes = config.get("kernel_sizes")  # [3, 3, 3]
        self.num_channels = config.get("num_channels")  # [32, 64, 128]
        self.drop_and_BN = config.get(
            "drop_and_BN"
        )  # 'BN-drop', 'drop-BN', 'BN-only', 'drop-only', 'none'
        self.num_layers = config.get("num_layers")  # int, e.g, 1
        self.num_heads = config.get("num_heads")  # int, e.g, 8
        self.model_dim = config.get("model_dim")
        self.dropout = config.get("dropout")  # number, e.g. 0.5

        self.train_bs = config.get("train_bs")
        self.test_bs = config.get("test_bs")
        self.momentum = config.get("momentum")
        self.epochs = config.get("epochs")

        self.ablation = config.get("ablation")
        self.dataset_name = config.get("dataset_name")  # weibo, twitter
        self.optimizer_name = config.get("optimizer_name")  # SGD, Adam, AdamW

        self.learning_rate = config.get("learning_rate")  # number
        self.weight_decay = config.get("weight_decay")  # number
        self.max_sen_len = config.get("max_sen_len")  # int
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.config_check()

        self.model = self.get_model().to(self.device)
        self.init_network()

        self.train_loader, self.test_loader = self.get_dataloader()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    def handle_batch_input(self, train_data):
        bert_paras = ["text_input_ids", "token_type_ids", "attention_mask"]
        vgg_paras = ["image"]
        dct_paras = ["dct_img"]
        share_paras = ['label', 'post_id']
        parameters = {}
        involve = bert_paras + vgg_paras + dct_paras
        involve += share_paras

        for para in involve:
            parameters[para] = train_data[para].to(self.device)

        return parameters

    def handle_model_input(self, parameters):
        outputs = self.model(parameters['text_input_ids'],
                             parameters['token_type_ids'],
                             parameters['attention_mask'],
                             parameters['image'],
                             parameters['dct_img'],
                             attn_mask=None)

        return outputs

    def train_one_time(self):
        # training
        loss_values, test_loss_values = [], []
        acc_values, test_acc_values = [], []
        test_precision_values = []
        test_recall_values = []
        test_f1_values = []

        for epoch_index, epoch in enumerate(range(self.epochs)):
            print('epoch:{}{}'.format(epoch_index, '-' * 20))

            self.model.train()

            train_batch_loss = []
            train_batch_acc = []
            for i, train_data in enumerate(self.train_loader):
                parameters = self.handle_batch_input(train_data)
                train_label = parameters['label']

                # Forward + Backward + Optimize
                self.model.zero_grad()
                outputs = self.handle_model_input(parameters)

                loss_input = outputs[0]  # after linear
                loss = self.criterion(loss_input, train_label)
                loss.backward(retain_graph=True)

                # Gradient cropping
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.max_grad_norm)

                train_label = train_label.cpu().detach().numpy().tolist()
                pred_input = torch.sigmoid(
                    outputs[1]).cpu().detach().numpy().tolist()  # output[1] is after linear+softmax

                acc = self.flat_accuracy(pred_input, train_label)
                self.optimizer.step()
                self.scheduler.step()

                train_batch_loss.append(loss.detach().item())
                train_batch_acc.append(acc)

            # Store the loss value for plotting the learning curve.
            train_epoch_loss = sum(train_batch_loss) / len(self.train_loader)
            loss_values.append(train_epoch_loss)

            # Store the acc value
            train_epoch_acc = sum(train_batch_acc) / len(self.train_loader)
            acc_values.append(train_epoch_acc)

            self.model.eval()

            test_batch_loss = []
            test_batch_acc = []
            report_label = []
            report_predict = []

            best_test_acc = 0

            for i, test_data in enumerate(self.test_loader):
                parameters = self.handle_batch_input(test_data)
                test_label = parameters['label']

                with torch.no_grad():
                    outputs = self.handle_model_input(parameters)

                test_loss_input = outputs[0]
                test_loss = self.criterion(test_loss_input, test_label)

                predict = torch.max(outputs[1].cpu().detach(), 1)[1]

                test_pred_input = torch.sigmoid(
                    outputs[1]).cpu().detach().numpy().tolist()  # output[1]
                test_label = test_label.cpu().detach().numpy().tolist()

                test_acc = self.flat_accuracy(test_pred_input, test_label)

                test_batch_loss.append(test_loss.detach().item())
                test_batch_acc.append(test_acc)

                for j in range(len(test_label)):
                    report_label.append(test_label[j])
                    report_predict.append(predict[j])

            test_epoch_loss = sum(test_batch_loss) / len(self.test_loader)
            test_epoch_acc = sum(test_batch_acc) / len(self.test_loader)

            report = classification_report(report_label, report_predict, output_dict=True)

            if test_epoch_acc > best_test_acc:
                best_test_acc = test_epoch_acc
                self.condition_save(epoch_index, test_epoch_acc, report)

            test_loss_values.append(test_epoch_loss)
            test_acc_values.append(test_epoch_acc)
            test_precision_values.append(float(report["macro avg"]["precision"]))
            test_recall_values.append(float(report["macro avg"]["recall"]))
            test_f1_values.append(float(report["macro avg"]["f1-score"]))

            self.print_result_table_handler(loss_values, acc_values, test_loss_values, test_acc_values,
                                            test_precision_values, test_recall_values, test_f1_values, report,
                                            print_type='tabel', table_type='pretty')

            # self.condition_save(epoch_index, test_epoch_acc, report)

            # early_stopping HERE～
            self.early_stopping(test_epoch_acc, test_recall_values)

            if self.early_stopping.early_stop:
                break

        return np.max(test_acc_values)

    def print_result_table_handler(self, loss_values, acc_values,
                                   test_loss_values, test_acc_values,
                                   test_precision_values, test_recall_values,
                                   test_f1_values, report, print_type='tabel',
                                   table_type='pretty'):

        # 这个地方需要绝对路径，不能使用相对路径
        if not os.path.exists('/home/xxxx/Reproduction_of_MCAN/results'):
            os.makedirs('/home/xxxx/Reproduction_of_MCAN/results')
        f_result = open(
            '/home/xxxx/Reproduction_of_MCAN/results/{}_accuracy.txt'.format(
                self.dataset_name), 'a+')

        def trend(values_list):
            if len(values_list) == 1:
                diff_value = values_list[-1]
                return '↑ ({:+.6f})'.format(diff_value)
            else:
                diff_value = values_list[-1] - values_list[-2]
                if values_list[-1] > values_list[-2]:
                    return '↑ ({:+.6f})'.format(diff_value)
                elif values_list[-1] == values_list[-2]:
                    return '~'
                else:
                    return '↓ ({:+.6f})'.format(diff_value)

        if print_type == 'tabel':
            avg_table = [["train loss", loss_values[-1], trend(loss_values)],
                         ["train acc", acc_values[-1], trend(acc_values)],
                         ["test loss", test_loss_values[-1], trend(test_loss_values)],
                         ["test acc", test_acc_values[-1], trend(test_acc_values)],
                         ["test pre", test_precision_values[-1], trend(test_precision_values)],
                         ['test rec', test_recall_values[-1], trend(test_recall_values)],
                         ['test F1', test_f1_values[-1], trend(test_f1_values)]]

            avg_header = ['metric', 'value', 'trend']
            print((tabulate(avg_table, avg_header, floatfmt=".6f", tablefmt=table_type)))

            class_table = [['0', report["0"]["precision"], report["0"]["recall"], report["0"]["f1-score"],
                            '{}/{}'.format(report["0"]["support"], report['macro avg']["support"])],
                           ['1', report["1"]["precision"], report["1"]["recall"], report["1"]["f1-score"],
                            '{}/{}'.format(report["1"]["support"], report['macro avg']["support"])]]

            class_header = ['class', 'precision', 'recall', 'f1', 'support']
            print((tabulate(class_table, class_header, floatfmt=".6f", tablefmt=table_type)))
        else:
            print(("Average train loss: {}".format(loss_values[-1])))
            print(("Average train acc: {}".format(acc_values[-1])))
            print(("Average test loss: {}".format(test_loss_values[-1])))
            print(("Average test acc: {}".format(test_acc_values[-1])))
            print(report)

        f_result.write(str(avg_header)+'\n')
        f_result.write(str(avg_table)+'\n')
        f_result.write(str(class_header)+'\n')
        f_result.write(str(class_table)+'\n')

        f_result.close()

    def step(self):
        test_acc = self.train_one_time()
        return {"best_test_accuracy": test_acc}

    def save_model(self, folder_path, epoch_index, test_acc, report):
        root = self.save_root
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

        path = os.path.join(root, folder_path)

        if not os.path.exists(path):
            os.makedirs(path)

        save_name = "task_{}-epoch_{}-model_{}-date-{}-acc_{}-precision_{}-recall_{}-f1_{}.pth".format(
            self.dataset_name, epoch_index, self.ablation, dt_string, test_acc, report["macro avg"]["precision"],
            report["macro avg"]["recall"], report["macro avg"]["f1-score"])
        print("Saving model to {}, as {}".format(path, save_name))

        state = {
            "net": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }
        torch.save(
            state,
            os.path.join(
                path,
                save_name,
            ),
        )

    def condition_save(self, epoch_index, test_epoch_acc, report):
        twitter_threshold = 0.8
        weibo_threshold = 0.89
        if self.dataset_name == 'twitter':
            if test_epoch_acc >= twitter_threshold:
                folder_path = 'model_save'
                self.save_model(folder_path, epoch_index, test_epoch_acc, report)
        elif self.dataset_name == 'weibo':
            if test_epoch_acc >= weibo_threshold:
                folder_path = 'model_save'
                self.save_model(folder_path, epoch_index, test_epoch_acc, report)

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


class AvgMetricCallback(Callback):

    def __init__(self):
        super(AvgMetricCallback, self).__init__()

    def init(self):
        try:
            self.results_df
            self.record_index += 1
        except:
            self.results_df = pd.DataFrame()
            self.record_index = 1

    def handle_parameters(self, config):
        for key, value in config.items():
            if isinstance(value, list):
                config[key] = str(value)

        df = pd.DataFrame(config, index=[self.record_index])
        return df

    def on_trial_complete(self, iteration, trials, trial, **info):
        self.init()

        config_df = self.handle_parameters(trial.config)
        config_df['trial'] = trial
        self.results_df = pd.concat([self.results_df, config_df], sort=False)


from ray.tune import CLIReporter
from ray.tune.experiment.trial import Trial
class TrialTerminationReporter(CLIReporter):
    def __init__(self):
        super(TrialTerminationReporter, self).__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated


if __name__ == '__main__':

    # grid search - weibo
    local_dir_root = './log/'
    repeat_times = 1
    max_concurrent = 1
    avg_metric = AvgMetricCallback()

    # tune.run是一个训练框架，可以实现自动化调参
    # 缺点是所有参数必须在这里输入，不能以参数的形式传入代码中
    analysis = tune.run(
        TrainALL,
        callbacks=[avg_metric],
        metric="best_test_accuracy",
        mode="max",
        name="twitter-experiment",
        local_dir=os.path.join(local_dir_root, '{}'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),
        resources_per_trial={"cpu": 1, "gpu": 1 / max_concurrent},
        stop={"training_iteration": 1},
        progress_reporter=TrialTerminationReporter(),
        num_samples=repeat_times,
        config={
            # 在这里对需要训练的数据集进行修改
            # "dataset_name": tune.grid_search(['weibo']),
            # "max_sen_len": tune.grid_search([160]),
            "dataset_name": tune.grid_search(['twitter']),
            "max_sen_len": tune.grid_search([25]),

            # 下面是修改模型的参数
            # Network
            # "ablation": tune.grid_search(
            #     ['bert', 'vgg', 'dct', 'bert+vgg+fusion', 'bert+dct+vgg+concat', 'bert+vgg+concat', "bert+dct+vgg+fusion"]),
            "ablation": tune.grid_search(["bert+dct+vgg+fusion"]),

            # "bert_model_name": tune.grid_search(["bert-base-chinese"]),
            "bert_model_name": tune.grid_search(["bert-base-cased"]),
            # "bert_model_name": tune.grid_search(["bert-base-multilingual-cased"]),
            "kernel_sizes": tune.grid_search([[3, 3, 3]]),
            "num_channels": tune.grid_search([[32, 64, 128]]),
            "num_layers": tune.grid_search([2]),
            "num_heads": tune.grid_search([4]),
            "dropout": tune.grid_search([0.5]),
            "drop_and_BN": tune.grid_search(['drop-BN']),  # 'drop-BN', 'BN-drop', 'BN-only', 'drop-only', 'none'
            "FREEZE_BERT": tune.grid_search([False]),
            "FREEZE_VGG": tune.grid_search([False]),
            "model_dim": tune.grid_search([256]),
            "init_method": tune.grid_search(['default']),

            # optimizer
            "optimizer_name": tune.grid_search(["AdaBelief"]),  # AdaBelief, Adam are better
            "learning_rate": tune.grid_search([0.0001]),
            # "bert_learning_rate": tune.loguniform(1e-5, 1e-2),
            # "vgg_learning_rate": tune.loguniform(1e-5, 1e-2),
            # "dtcconv_learning_rate": tune.loguniform(1e-5, 1e-2),
            # "fusion_learning_rate": tune.loguniform(1e-5, 1e-2),
            # "linear_learning_rate": tune.loguniform(1e-5, 1e-2),
            # "classifier_learning_rate": tune.loguniform(1e-5, 1e-2),
            "bert_learning_rate": tune.grid_search([0.0001]),
            "vgg_learning_rate": tune.grid_search([0.0001]),
            "dtcconv_learning_rate": tune.grid_search([0.0001]),
            "fusion_learning_rate": tune.grid_search([0.0001]),
            "linear_learning_rate": tune.grid_search([0.0001]),
            "classifier_learning_rate": tune.grid_search([0.0001]),

            "momentum": tune.grid_search([0.9]),
            "weight_decay": tune.grid_search([0.15]),
            "seed": tune.grid_search([43]),

            "early_stopping_patience": 10,

            # training
            # "epochs": tune.grid_search([100]),
            "epochs": tune.grid_search([1]),
            "train_bs": tune.grid_search([16]),
            "test_bs": tune.grid_search([16]),

        },
    )

    print("Best config is:", analysis.get_best_config(metric="best_test_accuracy", mode="max"))