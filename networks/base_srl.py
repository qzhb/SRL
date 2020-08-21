import torch
import torch.nn as nn
from torch.nn.init import normal, constant
from torch.nn import functional as F
from torch.nn import Parameter
from torch.nn import init
from torch import Tensor

import math
import numpy as np

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, qq, k, v):
        
        qq = qq.unsqueeze(1)
        
        sim = self.cos(qq.expand(k.size()), k)
        
        output = torch.matmul(sim.unsqueeze(1), v)
        
        output = output.squeeze(1)

        return output


class SelfAttention(nn.Module):

    def __init__(self, d_model, d_feature, dropout=0.5):
        super().__init__()
    
        self.trans = nn.Linear(d_model, d_feature)
        nn.init.normal_(self.trans.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_feature)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_feature, 0.5))

        self.dropout = nn.Dropout(dropout)

    def forward(self, query_feature, enc_feature):
        
        q = query_feature
        k = enc_feature
        v = enc_feature
        output = self.attention(q, k, v)

        return output

class EncGRU(nn.Module):
    def __init__(self, feat_in, feat_out, num_layers=1, dropout=0):
        super(EncGRU, self).__init__()

        self.gru = nn.GRU(feat_in, feat_out, num_layers=num_layers, dropout=dropout)

    def forward(self, seq):
        last_hid=None
        hid = []
        for i in range(seq.shape[0]):
            el = seq[i,...].unsqueeze(0)
            if last_hid is not None:
                _, last_hid = self.gru(el, last_hid)
            else:
                _, last_hid = self.gru(el)
            hid.append(last_hid)
           
        return torch.stack(hid, 0)

class Ant_Model(nn.Module):
    def __init__(self, num_class, verb_num_class, noun_num_class, feat_in, hidden, embedding_dim, S_enc, S_ant, ant_dropout_ratio=0.8, enc_dropout_ratio=0.8, clc_dropout_ratio=0.8, depth=1):
        super(Ant_Model, self).__init__()

        ##  parameters
        self.input_size = feat_in
        self.hidden_size = hidden
        self.verb_num_class = verb_num_class
        self.noun_num_class = noun_num_class
        self.embedding_dim = embedding_dim
        self.S_enc = S_enc
        self.S_ant = S_ant

        self.ant_dropout = nn.Dropout(ant_dropout_ratio)
        self.enc_dropout = nn.Dropout(enc_dropout_ratio)
        self.clc_dropout = nn.Dropout(clc_dropout_ratio)

        ## enc gru
        self.enc_gru = EncGRU(self.input_size, self.hidden_size, num_layers=depth)
        
        ## ant gru 
        self.input_size_a = self.hidden_size + self.input_size
        self.hidden_size_a = self.hidden_size
        self.ant_gru = nn.GRU(self.input_size_a, self.hidden_size_a, num_layers=1)

        ## reattend gru
        self.input_size_r = self.input_size + self.hidden_size_a
        self.hidden_size_r = self.hidden_size
        self.reattend_gru = nn.GRU(self.input_size_r, self.hidden_size_r, num_layers=1)
        
        reattend_gru_hidden = torch.Tensor(1, 1, self.hidden_size_r)
        nn.init.kaiming_normal_(reattend_gru_hidden, mode='fan_out', nonlinearity='relu')
        
        ## reattend
        self.reattend_net = SelfAttention(self.hidden_size_a, self.hidden_size)

        ## reinforce
        self.reinforce_net_verb = nn.Linear(self.hidden_size*2, self.verb_num_class)
        self.reinforce_net_noun = nn.Linear(self.hidden_size*2, self.noun_num_class)
        
        ## prediction
        self.classifier = nn.Linear(self.hidden_size*2, num_class)
        
        ## init parameters
        for l in self.children():
           if isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(l.bias, 0)

    def forward(self, inputs):

        inputs = inputs.permute(1, 0, 2) # batch * length * dim --> length * batch * dim
        temp_inputs = self.ant_dropout(F.relu(inputs))
        features = self.enc_gru(temp_inputs)
        features = features.squeeze(1).contiguous()  # length * batch * dim

        # accumulate the predictions in a list
        feature_predictions = []
        predictions = []

        # for each time-step
        for t in range(features.shape[0]):
            ## enc inputs
            enc_inputs = inputs[0:t+1, ...].permute(1, 0, 2) # batch * t * dim

            ## init input    
            i_t = torch.cat((inputs[t, ...], inputs[t, ...]), dim=1)
            i_t = self.enc_dropout(F.relu(i_t)).unsqueeze(0)
            h_t = features[t, ...].unsqueeze(0)
            
            h_t_re = features[t, ...].unsqueeze(0)
            
            ant_length = inputs.shape[0]-t
            for index in range(ant_length):
                ## ant gru
                # update gate
                _, h_t = self.ant_gru(i_t, h_t)
                
                ## reattend
                query_feature = h_t.squeeze(0)
                reattend_feature = self.reattend_net(query_feature, enc_inputs).squeeze(1)

                ## reattend GRU
                i_t_re = torch.cat((query_feature, reattend_feature), 1)
                i_t_re = self.enc_dropout(F.relu(i_t_re)).unsqueeze(0)
                # update gate
                _, h_t_re = self.reattend_gru(i_t_re, h_t_re)
                
                temp_h_t_re = h_t_re.squeeze(0)
                ## next inputs
                i_t = torch.cat((temp_h_t_re, inputs[t, ...]), dim=1)
                i_t = self.enc_dropout(F.relu(i_t)).unsqueeze(0)

                ## clc feature
                clc_feature = torch.cat((query_feature, temp_h_t_re), dim=1)
                 # accumulate
                if index < ant_length-1:
                    feature_predictions.append(query_feature)
                
            # accumulate 
            predictions.append(clc_feature)
        
        feature_predictions = torch.stack(feature_predictions, 1)
        x = torch.stack(predictions, 1)
        temp_x = x.view(-1, x.size(2))
        
        ## reinforce feature
        reinforce_feature = self.clc_dropout(F.relu(temp_x))
        reinforce_verb_p = self.reinforce_net_verb(reinforce_feature)
        reinforce_verb_p = reinforce_verb_p.view(x.size(0), x.size(1), -1)
       
        reinforce_noun_p = self.reinforce_net_noun(reinforce_feature)
        reinforce_noun_p = reinforce_noun_p.view(x.size(0), x.size(1), -1)
       
        ## classifier
        temp_x = self.clc_dropout(F.relu(temp_x))
        y = self.classifier(temp_x).view(x.size(0), x.size(1), -1)
        
        return y, feature_predictions, reinforce_verb_p, reinforce_noun_p

