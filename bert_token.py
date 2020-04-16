import os
import pickle
import sys

import numpy as np
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer

pretrained_path = 'scibert_scivocab_uncased'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')


def bert_sen_token(token_dict, traininstance, maxlen):
    tokenizer = Tokenizer(token_dict)
    train_indices = []
    train_segments = []
    train_text = []
    for text in traininstance:
        tokens = tokenizer.tokenize(text)
        indices, segments = tokenizer.encode(first=text, max_len=maxlen)
        train_indices.append(indices)
        train_segments.append(segments)
        train_text.append(tokens)

    return train_indices, train_segments,train_text


def bert_token(token_dict, traininstance, maxlen, maxsen):
    train_indices = np.zeros((len(traininstance),maxsen,maxlen))
    train_segments = np.zeros((len(traininstance),maxsen,maxlen))
    t_indices = []
    t_segments = []
    train_text = []
    for doc in traininstance:
        indices, segments, text = bert_sen_token(token_dict,doc,maxlen)
        t_indices.append(indices)
        t_segments.append(segments)
        train_text.append(text)

    for i in range(len(t_indices)):
        for j in range(len(t_indices[i])):
            train_indices[i][j]=t_indices[i][j]

    for i in range(len(t_segments)):
        for j in range(len(t_segments[i])):
            t_segments[i][j]=t_segments[i][j]

    return train_indices, train_segments,train_text


def load_pkl(name):
    file = open('./data/bert/' + name, 'rb')
    f = pickle.load(file)
    file.close()
    return f


def save_txt(name, data):
    f = open('./data/bert/' + name, 'w')
    # pickle.dump(data, f)
    for line in data:
        f.write(str(line))
        f.write("\n")
    f.close()


def save_file(name, data):
    f = open('./data/bert/' + name, 'wb')
    pickle.dump(data, f)
    f.close()


def get_entity1_pos(text,maxsen,maxlen):
    all_pos1 = np.zeros((len(text),maxsen,maxlen))

    for i in range(len(text)):
        for j in range(len(text[i])):
            start_index = -1
            end_index = -1

            for k in range(len(text[i][j])):
                if text[i][j][k] == 'ch':
                    if k + 1 < len(text[i][j]) and text[i][j][k + 1] == '_':
                        if k + 2 < len(text[i][j]) and text[i][j][k + 2] == 'start':
                            start_index=k
                            p=k+2
                            while p<len(text[i][j]):
                                if text[i][j][p] == 'ch':
                                    if p + 1 < len(text[i][j]) and text[i][j][p + 1] == '_':
                                        if p + 2 < len(text[i][j]) and text[i][j][p + 2] == 'end':
                                            end_index=p+2
                                            k=p+2
                                            break
                                p+=1
                            if start_index!= -1 and end_index!=-1:
                                q=start_index
                                while(q<=end_index):
                                    all_pos1[i][j][q]=1
                                    q=q+1

    return all_pos1


def get_entity2_pos(text,maxsen,maxlen):
    all_pos1 = np.zeros((len(text),maxsen,maxlen))

    for i in range(len(text)):
        for j in range(len(text[i])):
            start_index = -1
            end_index = -1

            for k in range(len(text[i][j])):
                if text[i][j][k] == 'ds':
                    if k + 1 < len(text[i][j]) and text[i][j][k + 1] == '_':
                        if k + 2 < len(text[i][j]) and text[i][j][k + 2] == 'start':
                            start_index=k
                            p=k+2
                            while p<len(text[i][j]):
                                if text[i][j][p] == 'ds':
                                    if p + 1 < len(text[i][j]) and text[i][j][p + 1] == '_':
                                        if p + 2 < len(text[i][j]) and text[i][j][p + 2] == 'end':
                                            end_index=p+2
                                            k=p+2
                                            break
                                p+=1
                            if start_index!= -1 and end_index!=-1:
                                q=start_index
                                while(q<=end_index):
                                    all_pos1[i][j][q]=1
                                    q=q+1

    return all_pos1

def get_position(entity_pos, maxlen, maxsentence):
    position = []
    print(entity_pos.shape)
    all_pos = np.reshape(entity_pos, [-1, maxlen*maxsentence])
    print(all_pos.shape)
    print(all_pos[0])
    all_pos = all_pos.tolist()
    tag=0
    for doc in all_pos:
        redoc=list(reversed(doc))
        sen_pos = []
        i = 0
        for sen in doc:
            if sen != 1:
                if 1 not in doc[i:]:
                    index1 = -1
                else:
                    index1 = doc[i:].index(1)
                if 1 not in redoc[maxlen * maxsentence - i - 1:]:
                    index2 = -1
                else:
                    index2 = redoc[maxlen * maxsentence - i - 1:].index(1)
                if(index1 == -1 and index2 == -1):
                    sen_pos.append(0)
                else:
                    index1 = index1 if index1!=-1 else sys.maxsize
                    index2 = index2 if index2!=-1 else sys.maxsize
                    pos = maxlen*maxsentence-index1 if index1<index2 else maxlen*maxsentence-index2
                    sen_pos.append(pos)
            else:
                sen_pos.append(maxlen*maxsentence)
            i+=1
        position.append(sen_pos)
        print("***************"+str(tag)+"*********")
        tag+=1
    position = np.array(position)
    position = np.reshape(position,[-1,maxsentence,maxlen])
    print(position.shape)
    return position

def para_comput(text):
    max_len=0
    max_sen=0
    for doc in text:
        max_sen=max(len(doc),max_sen)
        for sen in doc:
            max_len = max(len(sen), max_len)
    print("max_sen",max_sen)
    print("max_len", max_len)


if __name__ == "__main__":
    train_text = load_pkl("train_text")
    train_pos1 = load_pkl("train_pos1")
    train_pos2 = load_pkl("train_pos2")
    train_indices = load_pkl("train_indices")
    train_segments = load_pkl("train_segments")

    # maxsen = 23
    # maxlen = 452
    #
    # token_dict = load_vocabulary(vocab_path)
    # train_indices, train_segments,train_text = bert_token(token_dict, traininstance, maxlen,maxsen)
    # dev_indices, dev_segments,dev_text = bert_token(token_dict, devinstance, maxlen,maxsen)
    # test_indices, test_segments,test_text = bert_token(token_dict, testinstance, maxlen,maxsen)
    #
    # train_pos1 = get_entity1_pos(train_text, maxsen, maxlen)
    # dev_pos1 = get_entity1_pos(dev_text, maxsen, maxlen)
    # test_pos1 = get_entity1_pos(test_text, maxsen, maxlen)
    #
    # train_pos2 = get_entity2_pos(train_text, maxsen, maxlen)
    # dev_pos2 = get_entity2_pos(dev_text, maxsen, maxlen)
    # test_pos2 = get_entity2_pos(test_text, maxsen, maxlen)
    #
    # save_file("train_indices", train_indices)
    # save_file("train_segments", train_segments)
    # save_file("train_text", train_text)
    # save_file("dev_indices", dev_indices)
    # save_file("dev_segments", dev_segments)
    # save_file("dev_text", dev_text)
    # save_file("test_indices", test_indices)
    # save_file("test_segments", test_segments)
    # save_file("test_text", test_text)
    #
    # save_txt("1train_indices.txt", train_indices)
    # save_txt("1train_segments.txt", train_segments)
    # save_txt("1train_text.txt", train_text)
    # save_txt("1dev_indices.txt", dev_indices)
    # save_txt("1dev_segments.txt", dev_segments)
    # save_txt("1dev_text.txt", dev_text)
    # save_txt("1test_indices.txt", test_indices)
    # save_txt("1test_segments.txt", test_segments)
    # save_txt("1test_text.txt", test_text)
    #
    # save_file("train_pos1",train_pos1)
    # save_file("dev_pos1",dev_pos1)
    # save_file("test_pos1", test_pos1)
    # save_txt("1train_pos1.txt", train_pos1)
    # save_txt("1dev_pos1.txt", dev_pos1)
    # save_txt("1test_pos1.txt", test_pos1)
    #
    # save_file("train_pos2", train_pos2)
    # save_file("dev_pos2", dev_pos2)
    # save_file("test_pos2", test_pos2)
    # save_txt("1train_pos2.txt", train_pos2)
    # save_txt("1dev_pos2.txt", dev_pos2)
    # save_txt("1test_pos2.txt", test_pos2)
    print()