import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator,GNN,LocalAggregator_mix
from torch.nn import Module, Parameter
import torch.nn.functional as F


class CombineGraph(Module):
    def __init__(self, opt, num_total, n_category, category):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
 #      self.num_node = num_node
        self.num_total = num_total
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample

        
        self.n_category = n_category
        self.category = category
        # Aggregator
        self.local_agg_1 = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
       # self.local_agg_2 = LocalAggregator(50, self.opt.alpha, dropout=0.0)
        self.gnn = GNN(100)
        
        self.local_agg_mix_1 = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
   #    self.local_agg_mix_2 = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
   #     self.local_agg_mix_3 = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_total, self.dim)
        self.pos = nn.Embedding(200, self.dim)
        

        # Parameters_1
        self.w_1 = nn.Parameter(torch.Tensor(3 * self.dim, 2*self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(2*self.dim, 1))
        self.glu1 = nn.Linear(2*self.dim, 2*self.dim)
        self.glu2 = nn.Linear(2*self.dim, 2*self.dim, bias=False)
   #     self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)
        

        
 
       # self.aaa = Parameter(torch.Tensor(1))
        self.bbb = Parameter(torch.Tensor(1))
        self.ccc = Parameter(torch.Tensor(1))

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()
        
        item = []
        for x in range(1,num_total+1-n_category):
            item += [category[x]]
        item = np.asarray(item)  
        self.item =  trans_to_cuda(torch.Tensor(item).long())

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden1,hidden2,hidden1_mix,hidden2_mix, mask):
        hidden1 = hidden1 + hidden1_mix * self.bbb
        hidden2 = hidden2 + hidden2_mix * self.ccc
        hidden = torch.cat([hidden1, hidden2],-1)
        
        mask = mask.float().unsqueeze(-1)
        batch_size = hidden1.shape[0]
        len = hidden1.shape[1]
        
        pos_emb = self.pos.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        b = self.embedding.weight[1:self.num_total-self.n_category+1]  # n_nodes x latent_size
        item_category = self.embedding(self.item)     #n*d
        t = torch.cat([b,item_category],-1)

        scores = torch.matmul(select, t.transpose(1, 0))
        return scores

    def forward(self, inputs, adj, mask_item, item, items_ID, adj_ID, total_items, total_adj):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        
        hidden1 = self.embedding(inputs)
        hidden2 = self.embedding(items_ID)
        hidden_mix = self.embedding(total_items)
        
        # local
        hidden1 = self.local_agg_1(hidden1, adj, mask_item)
  #      hidden2 = self.local_agg_2(hidden2, adj_ID, mask_item)
        hidden2 = self.gnn(adj_ID,hidden2)
        
        hidden_mix = self.local_agg_mix_1(hidden_mix, total_adj, mask_item)
    #    hidden_mix = self.local_agg_mix_2(hidden_mix, total_adj, mask_item)
    #    hidden_mix = self.local_agg_mix_3(hidden_mix, total_adj, mask_item)

        # combine
        hidden1 = F.dropout(hidden1, self.dropout_local, training=self.training)
        hidden2 = F.dropout(hidden2, self.dropout_local, training=self.training)
        hidden_mix = F.dropout(hidden_mix, self.dropout_local, training=self.training)

        return hidden1, hidden2, hidden_mix


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    (alias_inputs, adj, items, mask, targets, inputs, alias_inputs_ID, adj_ID, items_ID, 
    alias_items, alias_category, total_adj, total_items)= data
    alias_items = trans_to_cuda( alias_items).long()
    alias_category = trans_to_cuda(alias_category).long()
    total_adj = trans_to_cuda(total_adj).float()
    total_items = trans_to_cuda(total_items).long()
    
    alias_inputs_ID = trans_to_cuda(alias_inputs_ID).long()
    items_ID = trans_to_cuda(items_ID).long()
    adj_ID = trans_to_cuda(adj_ID).float()
    
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()

    hidden1, hidden2, hidden_mix = model(items, adj, mask, inputs,  items_ID, adj_ID , total_items, total_adj)
    
    get1 = lambda i: hidden1[i][alias_inputs[i]]      #alias_inputs表示的是每个session中按序点击的商品在此session的“item”列表中所对应的相对位置
    seq_hidden1 = torch.stack([get1(i) for i in torch.arange(len(alias_inputs)).long()])
    get2 = lambda i: hidden2[i][alias_inputs_ID[i]]      #alias_inputs表示的是每个session中按序点击的商品在此session的“item”列表中所对应的相对位置
    seq_hidden2 = torch.stack([get2(i) for i in torch.arange(len(alias_inputs_ID)).long()])
    
    get1_mix = lambda i: hidden_mix[i][alias_items[i]]      #alias_inputs表示的是每个session中按序点击的商品在此session的“item”列表中所对应的相对位置
    seq_hidden1_mix = torch.stack([get1_mix(i) for i in torch.arange(len(alias_items)).long()])
    get2_mix = lambda i: hidden_mix[i][alias_category[i]]      #alias_inputs表示的是每个session中按序点击的商品在此session的“item”列表中所对应的相对位置
    seq_hidden2_mix = torch.stack([get2_mix(i) for i in torch.arange(len(alias_category)).long()])
    
    return targets, model.compute_scores(seq_hidden1,seq_hidden2,seq_hidden1_mix, seq_hidden2_mix, mask)


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()
    
    
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
#    hit, mrr = [], []
    hit_k10, mrr_k10, hit_k20, mrr_k20, hit_k30, mrr_k30, hit_k40, mrr_k40, hit_k50, mrr_k50 = [], [], [], [], [], [], [], [], [], []
    
    for data in test_loader:
        targets, scores = forward(model, data)
        sub_scores_k20 = scores.topk(20)[1]
        sub_scores_k20 = trans_to_cpu(sub_scores_k20).detach().numpy()
        sub_scores_k10 = scores.topk(10)[1]
        sub_scores_k10 = trans_to_cpu(sub_scores_k10).detach().numpy()
        
        sub_scores_k30 = scores.topk(30)[1]
        sub_scores_k30 = trans_to_cpu(sub_scores_k30).detach().numpy()
        sub_scores_k40 = scores.topk(40)[1]
        sub_scores_k40 = trans_to_cpu(sub_scores_k40).detach().numpy()
        sub_scores_k50 = scores.topk(50)[1]
        sub_scores_k50 = trans_to_cpu(sub_scores_k50).detach().numpy()
        targets = targets.numpy()
        
        for score, target, mask in zip(sub_scores_k20, targets, test_data.mask):
            hit_k20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k20.append(0)
            else:
                mrr_k20.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k10, targets, test_data.mask):
            hit_k10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k10.append(0)
            else:
                mrr_k10.append(1 / (np.where(score == target - 1)[0][0] + 1))
                
        for score, target, mask in zip(sub_scores_k30, targets, test_data.mask):
            hit_k30.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k30.append(0)
            else:
                mrr_k30.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k40, targets, test_data.mask):
            hit_k40.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k40.append(0)
            else:
                mrr_k40.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k50, targets, test_data.mask):
            hit_k50.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k50.append(0)
            else:
                mrr_k50.append(1 / (np.where(score == target - 1)[0][0] + 1))                
    
    result.append(np.mean(hit_k10) * 100)
    result.append(np.mean(mrr_k10) * 100)
    result.append(np.mean(hit_k20) * 100)
    result.append(np.mean(mrr_k20) * 100)
    
    result.append(np.mean(hit_k30) * 100)
    result.append(np.mean(mrr_k30) * 100)
    result.append(np.mean(hit_k40) * 100)
    result.append(np.mean(mrr_k40) * 100)
    result.append(np.mean(hit_k50) * 100)
    result.append(np.mean(mrr_k50) * 100)    

    return result