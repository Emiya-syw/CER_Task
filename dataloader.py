import jieba
import torch
from torch.utils.data import Dataset
import sys
import re
import random
import copy
from transformers import BertTokenizer,BertModel
import numpy as np
import tqdm
class dloader(Dataset):
    def __init__(self,mode,extractor,augment,file,MODELNAME='hfl/chinese-roberta-wwm-ext-large'):
        if file:
            if mode == 'train':
                self.path_train = "./data/1.txt"
            elif mode =='valid':
                self.path_train = "./data/1.txt"
                self.path_val = './data/2.txt'
        else:
            if mode == 'train':
                self.path_train = "./data/train.txt"
            elif mode =='valid':
                self.path_train = "./data/train.txt"
                self.path_val = './data/dev.txt'
            else: 
                sys.exit(0)
        self.mode = mode
        self.label2id = {'happy':0,'sad':1,'angry':2,'others':3}
        self.augment = augment
        self.augment_id = 0
        self.extractor = extractor
        if extractor == 'BERT':
            self.tokenizer = BertTokenizer.from_pretrained(MODELNAME)
            self.model = BertModel.from_pretrained(MODELNAME).cuda()
            self.data = self._bert_preprocess()
        else :
            self.data,self.num_vocab = self._cnn_preprocess()

    def _cnn_preprocess(self):
        data,split_data = self._split_data(self.path_train)
        print("词汇表拟合中...")
        voc = set(split_data)
        vocab = {}
        for idx,word in enumerate(voc):
            vocab[word] = idx
        if self.mode == 'valid':
            data,split_data = self._split_data(self.path_val)
            data = self._extractor_data(data,vocab)
        elif self.mode == 'train':
            data,split_data = self._split_data(self.path_train)
            data = self._extractor_data(data,vocab)
        elif self.mode == 'test':
            data,split_data = self._split_data(self.path_test)
            data = self._extractor_data(data,vocab)
        return data,len(voc)+1

    def _bert_preprocess(self):
        if self.mode=='train':
            data = self._split_data(self.path_train)
        elif self.mode =='valid':
            data = self._split_data(self.path_val)
        if self.augment:
            data = self._bert_augment(data)
        return data 
    
    def _split_data(self,path):
        data = {}
        split_data = []
        print("数据读取中...")
        with open(path,'r') as f:
            original_data = f.readlines()
            if self.extractor == 'BERT':
                with torch.no_grad():
                    for id,each in tqdm.tqdm(enumerate(original_data)):
                        sample = {} # sample = {'label':str,'A':list,'B':list,'C:list}
                        split_each = each.strip('\n')
                        split_each = split_each.split('\t')
                        sample['label'] = split_each[-1]
                        seqa = torch.LongTensor(self.tokenizer.encode(self._chinese(split_each[0]))).unsqueeze(0).cuda()
                        seqa = self.model(seqa)[1]
                        seqb = torch.LongTensor(self.tokenizer.encode(self._chinese(split_each[1]))).unsqueeze(0).cuda()
                        seqb = self.model(seqb)[1]
                        seqc = torch.LongTensor(self.tokenizer.encode(self._chinese(split_each[2]))).unsqueeze(0).cuda()
                        seqc = self.model(seqc)[1]
                        sample['sample']=torch.cat([seqa,seqb,seqc],dim=0)
                        data[id] = sample
                    return data
            else :
                for id,each in tqdm.tqdm(enumerate(original_data)):
                    sample = {} # sample = {'label':str,'A':list,'B':list,'C:list}
                    split_each = each.strip('\n')
                    split_each = split_each.split('\t')
                    sample['label'] = split_each[-1]
                    sample['A'] = [x for x in list(jieba.cut(self._chinese(split_each[0]))) if len(x)>0]
                    sample['B'] = [x for x in list(jieba.cut(self._chinese(split_each[1]))) if len(x)>0]
                    sample['C'] = [x for x in list(jieba.cut(self._chinese(split_each[2]))) if len(x)>0]
                    data[id] = sample
                    split_data += sample['A']
                    split_data += sample['B']
                    split_data += sample['C']
                return data,split_data

    def _extractor_data(self,data,vocab):
        print("特征提取生成中...")
        samples = []
        for id,each in enumerate(data.values()):
            data[id]['A'] = self._word2id(vocab,each['A']) 
            data[id]['B'] = self._word2id(vocab,each['B']) 
            data[id]['C'] = self._word2id(vocab,each['C'])
            if self.mode == 'train' and self.augment:
                if each['label'] == 'angry' or each['label'] == 'sad':
                    samples.append(self._cnn_augment(each,each['label'],len(vocab)))
        if self.mode == 'train' and self.augment:
            self.augment_id = len(data)
            for sample in samples:
                data[self.augment_id] = sample
                self.augment_id += 1
        return data

    def _cnn_augment(self,each,label,len_voc):
        sample = {}
        sample['label'] = label
        sample['A'] = copy.deepcopy(each['A'])
        sample['B'] = copy.deepcopy(each['B'])
        sample['C'] = copy.deepcopy(each['C'])
        lena = len(each['A'])
        lenb = len(each['B'])
        lenc = len(each['C'])
        if lena > 4:
            idx = random.randint(0,lena-1)
            id = random.randint(0,len_voc)
            sample['A'][idx] = id
        if lenb > 4:
            idx = random.randint(0,lenb-1)
            id = random.randint(0,len_voc)
            sample['B'][idx] = id
        if lenc > 4:
            idx = random.randint(0,lenc-1)
            id = random.randint(0,len_voc)
            sample['C'][idx] = id
        return sample
    
    def _bert_augment(self,data):
        augment_data = {}
        aug_num = 0
        for each in data.values():
            if each['label'] == 'angry' and each['label'] == 'sad':
                random_idx = np.random.randint(0,1024,(100,))
                for idx in random_idx:
                    if idx != 0:
                        each['sample'][0][idx] = each['sample'][0][idx-1]
                        each['sample'][1][idx] = each['sample'][1][idx-1]
                        each['sample'][2][idx] = each['sample'][2][idx-1]
                    else :
                        each['sample'][0][idx] = each['sample'][0][1023]
                        each['sample'][1][idx] = each['sample'][1][1023]
                        each['sample'][2][idx] = each['sample'][2][1023]
                augment_data[aug_num] = {}
                augment_data[aug_num]['label'] = each['label']
                augment_data[aug_num]['sample'] = each['sample']
        self.augment_id = len(data)
        for each in augment_data.values():
            data[self.augment_id] = each
            self.augment_id += 1
        return data

    def _word2id(self,vocab,seq):
        seq_id = []
        for word in seq:
            try:
                seq_id.append(vocab[word]+1)
            except:
                seq_id.append(0)
        return seq_id

    def _chinese(self,content):
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        chinese_txt = re.sub(pattern,'',content)
        if self.extractor == 'BERT' and len(chinese_txt)>510:
            chinese_txt = chinese_txt[0:510]
        return chinese_txt

    def _make_same_len(self,max_seq_len,seq):
        while(len(seq)<max_seq_len):
            seq.append(0)
        while(len(seq)>max_seq_len):
            seq.pop()
        return torch.Tensor(seq)

    def __getitem__(self,idx):
        label = torch.Tensor([self.label2id[self.data[idx]['label']]])
        umask = torch.Tensor([3])
        if self.extractor == 'CNN':
            seq_1 = self.data[idx]['A']
            seq_2 = self.data[idx]['B']
            seq_3 = self.data[idx]['C']
            max_seq_len = 100
            sample = torch.zeros(3,max_seq_len)
            sample[0] = self._make_same_len(max_seq_len,seq_1)
            sample[1] = self._make_same_len(max_seq_len,seq_2)
            sample[2] = self._make_same_len(max_seq_len,seq_3)
        else:
            sample = self.data[idx]['sample']
        return sample,label,umask
        
    def __len__(self):
        return len(self.data)
