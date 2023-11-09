# -*- coding: utf-8 -*-
import re
import os
from PIL import Image
import numpy as np

# 这里要用绝对路径，不能用相对路径，需要根据本机目录进行修改
data_path = './weibo_dataset'
new_train = './processd_data/weibo/train.txt'
new_test = './processd_data/weibo/test.txt'

# data_path = '/home/xxxx/Reproduction_of_MCAN/weibo_dataset'
# new_train = '/home/xxxx/Reproduction_of_MCAN/processd_data/weibo/train.txt'
# new_test = '/home/xxxx/Reproduction_of_MCAN/processd_data/weibo/test.txt'

image_file_list = [os.path.join(data_path, 'rumor_images/'), os.path.join(data_path, 'nonrumor_images/')]

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def read_images(file_list):
    image_list = {} 
    #img_num = 0
    for path in file_list:
        for filename in os.listdir(path):
            try:
                img = Image.open(path + filename).convert('RGB')
                img_id = filename.split('.')[0]
                image_list[img_id] = img
                #print('ok')
                #img_num += 1
            except:
                print(filename)
    return image_list#, img_num



def select_image(image_num, image_id_list, image_list):
    for i in range(image_num):
        #print('list:{}'.format(image_id_list))
        image_id = image_id_list[i]
        if image_id in image_list:
            #print('Yes, img_id:{}'.format(img_id))
            return image_id
    #f_log.write(line)
    return False            
            

def select_data(twitter_original_data, twitter_selected_data):
    """
    select features that we need from original data
    """
    f_old = open(twitter_original_data, 'r', encoding = 'UTF-8')
    # f_new = open(twitter_selected_data, 'w', encoding = 'UTF-8')
    f_new = open(twitter_selected_data, 'a+', encoding = 'UTF-8')

    fake_count = 0
    real_count = 0
    
    lines = f_old.readlines()
    for i, l in enumerate(lines):
        if i == 0:
            continue
        else:
            postId = l.split()[0]
            label = l.split()[-1]
            #print('postID:{}, label:{}'.format(postId, label))
                
            imageId_list = l.split('	')[4]
            postText = l.split('	')[1]
            
            clean_postText = re.sub(r"(http|https)((\W+)(\w+)(\W+)(\w*)(\W+)(\w*)|(\W+)(\w+)(\W+)|(\W+))","",postText)
            #print('clean_postText:{}'.format(clean_postText))
            
                
            if label == 'fake':
                label = '1'
                fake_count += 1
            elif label == 'real':
                label = '0'
                real_count += 1
            else:                    
                print('The label of this tweet is humor, we donnot need it.')
                
            f_new.write(postId + '|' + clean_postText + '|' + imageId_list + '|' + label + '\n')
                    
    f_old.close()
    f_new.close()
    
    return fake_count, real_count
     

def get_max_len(file):

    #Get the maximal length of sentence in dataset

    f = open(file, 'r', encoding = 'UTF-8')
    
    max_post_len = 0
    
    lines = f.readlines()
    post_num = len(lines)
    for i in range(post_num):
        post_content = list(lines[i].split('|')[1].split())
        tmp_len = len("".join(post_content))
        if tmp_len > max_post_len:
            max_post_len = tmp_len
            
    f.close()
    return max_post_len

def get_data(dataset, image_list):
    if dataset == 'train':        
        data_file = new_train
    else:
        data_file = new_test
        
    f = open(data_file, 'r', encoding = 'UTF-8')
    lines = f.readlines()
        
    data_post_id = []
    data_post_content = []
    data_image = []
    data_label = []   
        
    data_num = len(lines)
    unmatched_num = 0
        
    for line in lines:
        post_id = line.split('|')[0]
        post_content = line.split('|')[1]
        label = line.split('|')[-1].strip()

        # 这里还对所有image进行了选择，不需要了，本代码先选择了存在的image，再形成数据
        # image_id_list = line.split('|')[-2].strip().split(',')
        # #print(image_id_list)
        # img_num = len(image_id_list)
        # image_id = select_image(img_num, image_id_list, image_list)

        image_id = line.split('|')[-2].strip()
            
        if image_id != False:
            image = image_list[image_id]
                    
            data_post_id.append(int(post_id))
            data_post_content.append(post_content)
            data_image.append(image)
            data_label.append(int(label))
                    
        else:
            unmatched_num += 1
            continue
            
    f.close()
    
    data_dic = {'post_id': np.array(data_post_id),
                'post_content': data_post_content,
                'image': data_image,
                'label': np.array(data_label)
            }
    return data_dic, data_num-unmatched_num              


if __name__ == '__main__':

    for dtype in ['train', 'test']:
        for rtype in ['rumor', 'nonrumor']:
            original_train_data = os.path.join(data_path, 'tweets/{}_{}.txt'.format(dtype, rtype))

            makedir('./processd_data/weibo')
            new_train = './processd_data/weibo/{}.txt'.format(dtype)

            # 处理weibo原始数据，挑选模型需要的data写入本目录下新的train.txt文件中
            fake_num, real_num = select_data(original_train_data, new_train)
            print(fake_num, real_num)
            max_len = get_max_len(new_train)
            print(max_len)
            # f_log = open(log, 'w', encoding = 'UTF-8')

    # image_file_list = [os.path.join(data_path, 'rumor_images/'), os.path.join(data_path, 'nonrumor_images/')]

    # img_list = read_images(image_file_list)
    # print(img_num)
    # train, train_num = get_data('train', img_list)
    # print(train_num)











