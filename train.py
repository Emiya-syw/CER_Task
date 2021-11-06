
import numpy as np
import argparse, time
from sklearn.utils import validation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from model import BERT_LSTMModel,TriC_LSTMModel ,MaskedNLLLoss
from dataloader import dloader
import os 
import random
import matplotlib.pyplot as plt

def get_loaders(extractor,augment,file,batch_size=32, num_workers=0, pin_memory=False):
    
    trainset = dloader('train',extractor,augment,file)
    validset = dloader('valid',extractor,augment,file)
    
    train_loader = DataLoader(trainset,
                              shuffle= True,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    if extractor == 'BERT':
        return train_loader,valid_loader
    else: 
        return train_loader,trainset.num_vocab,valid_loader
        

def train_or_eval_model(model, loss_function, dataloader, epoch, extractor,optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    # one epoch
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        input_sequence, label ,umask= data
        input_sequence = input_sequence.cuda()
        label = label.cuda()
        umask = umask.cuda()

        log_prob= model(input_sequence, umask)
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])   #(batch_size,4) 
        labels_ = label.view(-1)
        if extractor == 'CNN':
            loss = loss_function(lp_, labels_, umask)
        else:
            loss = loss_function(lp_, labels_.long())
        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        losses.append(loss.item())
        if train:
            loss.backward()
            optimizer.step()
    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses)/len(losses),4)
    avg_accuracy = round(accuracy_score(labels,preds)*100,2)
    avg_fscore = round(f1_score(labels,preds,average='micro', labels=[0,1,2])*100,2)
    happy_fscore = round(f1_score(labels,preds,average='micro', labels=[0])*100,2)
    sad_fscore = round(f1_score(labels,preds,average='micro', labels=[1])*100,2)
    angry_fscore = round(f1_score(labels,preds,average='micro', labels=[2])*100,2)
    print(f'happy:{happy_fscore},sad:{sad_fscore},angry:{angry_fscore}')
    return avg_loss, avg_accuracy, labels, preds,avg_fscore

if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--model', type=str, default='BERT')
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--file', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=4, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weight')
    parser.add_argument('--cnn_filters', type=int, default=50, help='Number of CNN filters for feature extraction')
    parser.add_argument('--cnn_output_size', type=int, default=100, help='Output feature size from CNN layer')
    parser.add_argument('--cnn_dropout', type=float, default=0.5, metavar='cnn_dropout', help='CNN dropout rate')
    args = parser.parse_args()

    print(args)

    # cuda
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    # basic information
    batch_size = args.batch_size
    cuda       = args.cuda
    n_epochs   = args.epochs
    n_classes  = 4
    if args.model == 'CNN':
        D_e = 100
        D_h = 100
    else:
        D_e = 512
        D_h = 512
        
    kernel_sizes = [3,4,5]
    if args.model=='BERT':
        train_loader,valid_loader= get_loaders(args.model,args.augment,args.file,batch_size=batch_size,num_workers=0)
    else:
        train_loader,vocab_size,valid_loader= get_loaders(args.model,args.augment,args.file,batch_size=batch_size,num_workers=0)

    loss_weights = torch.FloatTensor([0.6149, 2.8586, 2.6742, 0.2717]).cuda()

    if args.model == 'BERT':
        model = BERT_LSTMModel(D_e, D_h,
                            n_classes=n_classes,
                            dropout=args.dropout,
                            )
        loss_function = nn.CrossEntropyLoss()
    else:
        model = TriC_LSTMModel(D_e, D_h,
                            vocab_size=vocab_size,
                            cnn_output_size=args.cnn_output_size,
                            cnn_filters=args.cnn_filters, 
                            cnn_kernel_sizes=kernel_sizes,
                            cnn_dropout=args.cnn_dropout,
                            n_classes=n_classes,
                            dropout=args.dropout,
                            )
        if args.class_weight:
            loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            loss_function = MaskedNLLLoss()
    
    if cuda:
        model.cuda()
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr,
                           weight_decay=args.l2)

    best_loss, best_label, best_pred, best_mask = None, None, None, None
    best_val_fscore = 0
    best_epoch = 0
    best_val_acc = 0
    x = [i for i in range(1,51)]
    y = []

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _,train_fscore = train_or_eval_model(model, loss_function,
                                               train_loader, e,args.model, optimizer, True)
        print('epoch {} train_loss {} train_acc {} train_fscore {}'.format(e+1, train_loss, train_acc, train_fscore))
        y.append(train_loss)

        valid_loss, valid_acc, _, _, val_fscore= train_or_eval_model(model, loss_function, valid_loader, e,args.model)
        if val_fscore > best_val_fscore:
            best_val_fscore = val_fscore
            best_val_acc = valid_acc
            best_epoch = e + 1
            name = str(best_epoch)+'_'+str(best_val_fscore)+'_'+'BERT_LSTM.pth'
            torch.save(model,name)
        print('epoch {} valid_loss {} valid_acc {} valid_fscore {}'.format(e+1, valid_loss, valid_acc, val_fscore))

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x,y)
    ax.set_xlabel('epoch')
    ax.set_ylabel('train_loss')
    title = 'BERT+1x1+L2+augment'
    ax.set_title(title)
    path = title + '.png'
    plt.savefig(path)
    print(f'best_val_f1score:{best_val_fscore} in epoch:{best_epoch} and accuracy is {best_val_acc}')
