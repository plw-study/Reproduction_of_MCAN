import os
import tqdm
from PIL import Image
import re
# import data_process_weibo as pro
# import data_process_twitter as pro_t


def get_weibo_matrix(data_type):

    corpus_dir = './weibo_dataset'

    if not os.path.exists('./processd_data/weibo'):
        os.makedirs('./processd_data/weibo')
    f_new = open('./processd_data/weibo/{}.txt'.format(data_type), 'a+')

    if data_type not in ['train', 'test']:
        raise ValueError('ERROR! data type must be train or test.')
    rumor_content = open('{}/tweets/{}_rumor.txt'.format(corpus_dir, data_type), 'r').readlines()
    nonrumor_content = open('{}/tweets/{}_nonrumor.txt'.format(corpus_dir, data_type), 'r').readlines()
    rumor_images = os.listdir('{}/rumor_images/'.format(corpus_dir))
    nonrumor_images = os.listdir('{}/nonrumor_images/'.format(corpus_dir))

    # text_lists = []  # [train_number]
    # image_lists = []  # [train_num]
    # labels = []  # [train_num, 2]
    # text_image_ids = []  # tweet_id|img_id

    n_lines = len(rumor_content)
    for idx in tqdm.tqdm(range(2, n_lines, 3)):
    # for idx in range(2, n_lines, 3):
        postId = rumor_content[idx-2].split('|')[0]
        postText = rumor_content[idx].strip()
        # one_rumor = text_filter_chinese(one_rumor)
        clean_postText = re.sub(r"(http|https)((\W+)(\w+)(\W+)(\w*)(\W+)(\w*)|(\W+)(\w+)(\W+)|(\W+))", "", postText)
        if postText:
            images = rumor_content[idx-1].split('|')
            for image in images:
                img = image.split('/')[-1]
                if img in rumor_images:
                    # image_lists.append(img)
                    # im = '{}/rumor_images/{}'.format(corpus_dir, img)
                    im = img.split('.')[0]
                    # im = Image.open(im_path, 'r').convert('RGB')
                    # image_lists.append(picture_filter())
                    # labels.append([0, 1])
                    label = '1'
                    # text_tokens, text_seqs = tokenizer.encode(first=one_rumor, max_len=seq_len)
                    # while len(text_tokens) < seq_len:
                    #     text_tokens.append(0)
                    # results = bert_model.predict([np.array([text_tokens]), np.array([text_seqs])])[0]
                    # text_tokens_matrix.append(results.tolist())
                    # text_lists.append(one_rumor)
                    # text_image_ids.append('{}|{}'.format(tweet_id, img.split('.')[0]))

                    f_new.write(postId + '|' + clean_postText + '|' + im + '|' + label + '\n')

                    # break

    n_lines = len(nonrumor_content)
    for idx in tqdm.tqdm(range(2, n_lines, 3)):
    # for idx in range(2, n_lines, 3):
        postId = nonrumor_content[idx-2].split('|')[0]
        postText = nonrumor_content[idx].strip()
        # one_nonrumor = text_filter_chinese(one_nonrumor)
        clean_postText = re.sub(r"(http|https)((\W+)(\w+)(\W+)(\w*)(\W+)(\w*)|(\W+)(\w+)(\W+)|(\W+))", "", postText)
        if postText:
            images = nonrumor_content[idx-1].split('|')
            for image in images:
                img = image.split('/')[-1]
                if img in nonrumor_images:
                    # image_lists.append(img)
                    # im = '{}/rumor_images/{}'.format(corpus_dir, img)
                    im = img.split('.')[0]
                    # im = Image.open(im_path, 'r').convert('RGB')
                    # image_lists.append(picture_filter('{}/nonrumor_images/{}'.format(corpus_dir, img)))
                    # labels.append([1, 0])
                    label = '0'
                    # text_tokens, text_seqs = tokenizer.encode(
                    #       first=one_rumor, max_len=seq_len)  # get the ids of tokens in a rumor
                    # while len(text_tokens) < seq_len:
                    #     text_tokens.append(0)
                    # results = bert_model.predict([np.array([text_tokens]), np.array([text_seqs])])[0]
                    # text_tokens_matrix.append(results.tolist())
                    # text_lists.append(one_nonrumor)
                    # text_image_ids.append('{}|{}'.format(tweet_id, img.split('.')[0]))

                    f_new.write(postId + '|' + clean_postText + '|' + im + '|' + label + '\n')

                    # break

    # assert len(text_lists) == len(image_lists) == len(labels) == len(text_image_ids)
    # print('   {} samples in {} set.'.format(len(labels), data_type))
    # print('   type of text list {}, image lists {}, labels {}, text image ids {}'.format(
    #     type(text_lists), type(image_lists), type(labels), type(text_image_ids),
    # ))

    f_new.close()

    return


def get_twitter_matrix(data_type):
    text_lists = []  # [train_number]
    image_lists = []  # [train_num]
    labels = []  # [train_num, 2]
    # label_dict = {'fake': [0, 1], 'real': [1, 0]}
    label_dict = {'fake': '1', 'real': '0'}
    text_image_ids = []

    if not os.path.exists('./processd_data/twitter'):
        os.makedirs('./processd_data/twitter')
    f_new = open('./processd_data/twitter/{}.txt'.format(data_type), 'a+')

    corpus_dir = './twitter_dataset'
    if data_type == 'train':
        tweets = open('{}/devset/posts.txt'.format(corpus_dir), 'r').readlines()[1:]
        image_index = 3
        image_dirs = '{}/devset/images/'.format(corpus_dir)
        image_files = list(filter(lambda x: not x.endswith('.txt'), os.listdir(image_dirs)))
        # 因为twitter里面post.txt中记录的是不带格式的图像文件名称，所以找图片的时候需要对应好有无格式.jpg
        image_name = [image_file.split('.')[0] for image_file in image_files]
    elif data_type == 'test':
        tweets = open('{}/testset/posts_groundtruth.txt'.format(corpus_dir), 'r').readlines()[1:]
        image_index = 4
        image_dirs = '{}/testset/images/'.format(corpus_dir)
        image_files = list(filter(lambda x: not x.endswith('.txt'), os.listdir(image_dirs)))
        image_name = [image_file.split('.')[0] for image_file in image_files]
    else:
        raise ValueError('data type must be train or test!')

    for lines in tqdm.tqdm(tweets):
    # for lines in tweets:
        args = lines.strip().split('\t')
        postId = args[0]
        for img in args[image_index].split(','):
            if img in image_name:
                # image_lists.append(image_files[image_name.index(img)])
                # im = '{}/{}'.format(image_dirs, image_files[image_name.index(img)])
                im = img
                # image_lists.append(picture_filter())
                # labels.append(label_dict[args[-1]])
                # event_labels.append(event_dict[img.split('_')[0]])
                label = label_dict[args[-1]]
                postText = args[1]
                # if tweet_id in translated_dict:
                #     tweet_text = translated_dict[args[0]]
                # tweet_text = text_filter_english(tweet_text)
                clean_postText = re.sub(r"(http|https)((\W+)(\w+)(\W+)(\w*)(\W+)(\w*)|(\W+)(\w+)(\W+)|(\W+))", "",
                                        postText)
                # text_tokens, text_seqs = tokenizer_input.encode(first=tweet_text, max_len=seq_len)
                # while len(text_tokens) < seq_len:
                    # text_tokens.append(0)
                # results = bert_model.predict([np.array([text_tokens]), np.array([text_seqs])])[0]
                # text_tokens_matrix.append(results.tolist())
                # text_lists.append(tweet_text)
                # text_image_ids.append('{}|{}'.format(tweet_id, img))

                f_new.write(postId + '|' + clean_postText + '|' + im + '|' + label + '\n')
                # break

    # assert len(text_lists) == len(image_lists) == len(labels) == len(text_image_ids)
    # print('   {} samples in {} set.'.format(len(labels), data_type))
    # print('   type of text list {}, image lists {}, labels {}, event labels {}, text image ids {}'.format(
    #     type(text_lists), type(image_lists), type(labels), type(event_labels), type(text_image_ids),
    # ))

    f_new.close()

    return


if __name__ == '__main__':
    print('===> 1. process weibo dataset......')
    if os.path.exists('./processd_data/weibo/train.txt'):
        os.remove('./processd_data/weibo/train.txt')
    if os.path.exists('./processd_data/weibo/test.txt'):
        os.remove('./processd_data/weibo/test.txt')
    get_weibo_matrix('train')
    get_weibo_matrix('test')

    print('===> 2. process twitter dataset......')
    if os.path.exists('./processd_data/twitter/train.txt'):
        os.remove('./processd_data/twitter/train.txt')
    if os.path.exists('./processd_data/twitter/test.txt'):
        os.remove('./processd_data/twitter/test.txt')
    get_twitter_matrix('train')
    get_twitter_matrix('test')

    # image_list = pro.read_images(pro.image_file_list)
    # data, data_numb = pro.get_data('train', image_list)
    # print(data_numb)

    # image_list = pro_t.read_images(pro_t.image_file_list)
    # print('len of image list:', len(image_list))
    # data, data_numb = pro_t.get_data('test', image_list)
    # print(data_numb)
