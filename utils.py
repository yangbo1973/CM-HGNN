import numpy as np
import torch
from torch.utils.data import Dataset


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len


def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity

def pro_inputs(category,inputs):   #为种类设置特征变量，先读出每个序列对应的种类ID序列
    inputs_ID = []
    for item in inputs:
       if item == 0:
          inputs_ID += [0]
       else:
          inputs_ID += [category[item]]
    return inputs_ID 

class Data(Dataset):
    def __init__(self, data,category, train_len=None):
        inputs, mask, max_len = handle_data(data[0], train_len)
        self.category = category
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len

    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]
        input_ID = pro_inputs(self.category,u_input)
        total = np.append(u_input,input_ID)
        total = total[total > 0]

        max_n_node = self.max_len
        node = np.unique(u_input)
        node_ID = np.unique(input_ID)
        total_node = np.unique(total)
        if len(total_node)<max_n_node*2:
          total_node= np.append(total_node,0)
          
        items = node.tolist() + (max_n_node - len(node)) * [0]
        items_ID = node_ID.tolist()  + (max_n_node - len(node_ID)) * [0]
        total_items = total_node.tolist() + (max_n_node*2 - len(total_node)) * [0]
        
        adj = np.zeros((max_n_node, max_n_node))
        u_A = np.zeros((max_n_node, max_n_node))
        total_adj = np.zeros((max_n_node*2, max_n_node*2))
        
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]
        
        for i in np.arange(len(u_input) - 1):
            u = np.where(total_node == u_input[i])[0][0]
            c = np.where(total_node == self.category[u_input[i]])[0][0]
            total_adj[u][u] = 1
            total_adj[c][c] = 4
            total_adj[u][c]= 2
            total_adj[c][u]= 3
            if u_input[i + 1] == 0:
                break          
            u2 = np.where(total_node == u_input[i + 1])[0][0]
            c2 = np.where(total_node == self.category[u_input[i + 1]])[0][0]
            total_adj[u][u2] = 1
            total_adj[u2][u] = 1
            
            total_adj[c][c2] = 4
            total_adj[c2][c] = 4


        alias_items = [np.where(total_node == i)[0][0] for i in u_input]
        alias_category = [np.where(total_node == i)[0][0] for i in input_ID]   #对应ID的相对位置
        
        for i in np.arange(len(input_ID) - 1):
            if input_ID[i + 1] == 0:
                break
            u = np.where(node_ID == input_ID[i])[0][0]
            v = np.where(node_ID == input_ID[i + 1])[0][0]
            u_A[u][v] += 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        alias_inputs_ID = [np.where(node_ID == i)[0][0] for i in input_ID]
        

        
        return [torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input),
                torch.tensor(alias_inputs_ID), torch.tensor(u_A), torch.tensor(items_ID),
                torch.tensor(alias_items),torch.tensor(alias_category),torch.tensor(total_adj),torch.tensor(total_items)]

    def __len__(self):
        return self.length
