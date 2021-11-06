import math
import torch
from torch.functional import _return_output
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,BertConfig


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        target = target.type(LongTensor)
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss
    
class CNNFeatureExtractor(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, output_size, filters, kernel_sizes, dropout):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=K) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
        self.feature_dim = output_size


    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        # if is_static:
        self.embedding.weight.requires_grad = False


    def forward(self, x, umask):
        x = x.transpose(1,0).contiguous()
        num_utt, batch, num_words = x.size()
        x = x.type(torch.LongTensor)  # (num_utt, batch, num_words)
        x = x.view(-1, num_words) # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
        x = x.cuda()
        emb = self.embedding(x) # (num_utt * batch, num_words) -> (num_utt * batch, num_words, 300) 
        emb = emb.transpose(-2, -1).contiguous() # (num_utt * batch, num_words, 300)  -> (num_utt * batch, 300, num_words) 
        convoluted = [F.relu(conv(emb)) for conv in self.convs] 
        #print(convoluted[0].shape,len(convoluted)) 
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted] #(num_utt * batch, 50) 
        #print(pooled[0].shape)
        concated = torch.cat(pooled, 1)
        features = F.relu(self.fc(self.dropout(concated))) # (num_utt * batch, 150) -> (num_utt * batch, 100)
        features = features.view(num_utt, batch, -1) # (num_utt * batch, 100) -> (num_utt, batch, 100)
        mask = umask.unsqueeze(-1).type(torch.FloatTensor) # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1) # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, self.feature_dim) #  (num_utt, batch, 1) -> (num_utt, batch, 100)
        mask = mask.cuda()
        features = (features * mask) # (num_utt, batch, 100) -> (num_utt, batch, 100)
        return features

class BERT_LSTMModel(nn.Module):

    def __init__(self, D_e, D_h,
                 n_classes=7, dropout=0.5):
        
        super(BERT_LSTMModel, self).__init__()
        self.dropout   = nn.Dropout(dropout)
        self.conv = nn.Conv1d(1024,D_e,1)
        self.fusion = nn.Linear(2*D_h,D_h)
        self.lstm = nn.LSTM(input_size=D_e, hidden_size=D_e, num_layers=2, bidirectional=False, dropout=dropout)
        self.linear = nn.Linear(D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def context(self,U):
        target = U[-1,:,:]
        target = target.unsqueeze(0)
        target = torch.cat([target,target,target],0)
        U = torch.cat([U,target],dim=2)
        U = self.fusion(U)
        return U

    def conv_1x1(self,U):
        U = U.transpose(2,1)
        U = F.relu(self.conv(U))
        U = U.transpose(2,1) # (3,8,512)
        return U

    def forward(self, input_seq, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = input_seq.transpose(1,0)
        U = self.conv_1x1(U)
        #U = self.context(U)
        emotions, (hidden,_) = self.lstm(U) # emotions:(3,8,1024)
        emotions = F.relu(self.linear(emotions[-1]))# (8,1024) -> (8,512)
        hidden = self.smax_fc(emotions).unsqueeze(0) # (1,8,4)
        log_prob = F.log_softmax(hidden, 2) 
        return log_prob
    
class TriC_LSTMModel(nn.Module):
    def __init__(self, D_e, D_h,
                 vocab_size, embedding_dim=300, 
                 cnn_output_size=100, cnn_filters=50, cnn_kernel_sizes=(3,4,5), cnn_dropout=0.5,
                 n_classes=7, dropout=0.5):
        super(TriC_LSTMModel, self).__init__()
        self.cnn_feat_extractor = CNNFeatureExtractor(vocab_size, embedding_dim, cnn_output_size, cnn_filters, cnn_kernel_sizes, cnn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=D_e, num_layers=1, bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)
        self.fusion = nn.Linear(2*D_h,D_h)

    def forward(self, input_seq, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.cnn_feat_extractor(input_seq, umask)

        target = U[-1,:,:]
        target = target.unsqueeze(0)
        target = torch.cat([target,target,target],0)
        U = torch.cat([U,target],dim=2)
        U = self.fusion(U)

        emotions, hidden = self.lstm(U) # emotion:(seq_len, batch, num_directions * hidden_size) hidden:(num_layers * num_directions, batch, hidden_size)
        hidden = F.elu(self.linear(emotions))
        
        hidden = self.dropout(hidden)
        hidden = hidden[-1,:,:]
        hidden = hidden.unsqueeze(0)
        hidden = self.smax_fc(hidden)
        log_prob = F.log_softmax(hidden, 2)
        return log_prob