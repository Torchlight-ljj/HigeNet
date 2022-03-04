
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import global_var
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import GPUtil
import psutil
from torchstat import stat
from thop import profile
from torchsummary import summary
class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries,keys,values,attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, flag = False, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        #ours adding
        self.our_flag = flag
        if self.our_flag:
            self.prob_con = nn.Conv1d(in_channels=64,out_channels=1,kernel_size=3,padding=1)
            nn.init.kaiming_normal_(self.prob_con.weight,mode='fan_in',nonlinearity='leaky_relu')
            self.relu = nn.ReLU()
        # self.cos_sim = None

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        #prob_nn
        if self.our_flag:
            # print(self.prob_con.kernel_size)
            q_k_out =(self.prob_con(torch.flatten(Q.permute(0,1,3,2),1,2)) + self.prob_con(torch.flatten(K.permute(0,1,3,2),1,2)))
            # print(q_k_out.shape)
            M = self.relu(q_k_out)
            M = nn.Softmax(dim=2)(M)
            # print(M.shape)
            M_top = M.topk(n_top,sorted = False)[1]
        else:
            K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
            index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
            K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
            Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

            # find the Top_k query with sparisty measurement
            M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
            M_top = M.topk(n_top, sorted=False)[1]            

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            print(V_sum.shape)
            contex = torch.zeros_like(V)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


def performance(batch):
    times0 = []
    times1 = []
    times2 = []
    gpus_utils =[[],[],[]]
    mem_utils =[[],[],[]]
    start = 10
    end = 1000
    x = np.linspace(start,end,end-start-1) 
    for length in range(start,end):
        full = FullAttention().cuda()
        Q = torch.ones((batch,length,8,64)).cuda()
        K = torch.ones((batch,length,8,64)).cuda()
        V = torch.ones((batch,length,8,64)).cuda()
        t0 = time.time()
        y = full(Q,K,V,None)
        t1 = time.time()
        gpu = GPUtil.getGPUs()[0]
        mem = psutil.virtual_memory()
        gpus_utils[0].append(round(gpu.memoryUsed/1024,3))
        mem_utils[0].append(round(mem.used/1024/1024,3))  
        waste = round((t1 - t0),8)
        times0.append(waste)
        print("Length:{},Time:{:.8f}s.".format(length,waste))
        torch.cuda.empty_cache()

    for length in range(start,end):
        prob = ProbAttention(False).cuda()
        Q = torch.ones((batch,length,8,64)).cuda()
        K = torch.ones((batch,length,8,64)).cuda()
        V = torch.ones((batch,length,8,64)).cuda()
        t0 = time.time()
        y = prob(Q,K,V,None)
        t1 = time.time()
        gpu = GPUtil.getGPUs()[0]
        mem = psutil.virtual_memory()
        gpus_utils[1].append(round(gpu.memoryUsed/1024,3))
        mem_utils[1].append(round(mem.used/1024/1024,3))  
        waste = round((t1 - t0),8)
        times1.append(waste)
        print("Length:{},Time:{:.8f}s.".format(length,waste))
        torch.cuda.empty_cache()

    for length in range(start,end):
        ourProb = ProbAttention(True).cuda()
        Q = torch.ones((batch,length,8,64)).cuda()
        K = torch.ones((batch,length,8,64)).cuda()
        V = torch.ones((batch,length,8,64)).cuda()
        t0 = time.time()
        y = ourProb(Q,K,V,None)
        t1 = time.time()
        gpu = GPUtil.getGPUs()[0]
        mem = psutil.virtual_memory()
        gpus_utils[2].append(round(gpu.memoryUsed/1024,3))
        mem_utils[2].append(round(mem.used/1024/1024,3))  
        waste = round((t1 - t0),8)
        times2.append(waste)
        print("Length:{},Time:{:.8f}s.".format(length,waste))
        torch.cuda.empty_cache()


    times0 = np.array(times0)
    times1 = np.array(times1)
    times2 = np.array(times2)
    p1, = plt.plot(x,times0[1:],color='blue',linewidth=1,label='Canonical')
    p2, = plt.plot(x,times1[1:],color='red',linewidth=1,label='ProbSparse')
    p3, = plt.plot(x,times2[1:],color='green',linewidth=1,label='NeuralSparse')
    plt.xlabel("Time steps")
    plt.ylabel("Time/s")
    plt.legend([p1,p2,p3], ["Canonical","ProbSparse","NeuralSparse"], loc='upper left')
    plt.savefig('./performance/'+str(batch)+'/comps.png')
    plt.close('all')

    x = np.linspace(start,end,end-start) 
    gpus_utils = np.array(gpus_utils)

    p1, = plt.plot(x,gpus_utils[0],color='blue',linewidth=1,label='Canonical')
    p2, = plt.plot(x,gpus_utils[1],color='red',linewidth=1,label='ProbSparse')
    p3, = plt.plot(x,gpus_utils[2],color='green',linewidth=1,label='NeuralSparse')
    plt.xlabel("Time steps")
    plt.ylabel("gpu")
    plt.legend([p1,p2,p3], ["Canonical","ProbSparse","NeuralSparse"], loc='upper left')
    plt.savefig('./performance/'+str(batch)+'/gpu_comps.png')
    plt.close('all')

    mem_utils = np.array(mem_utils)
    p1, = plt.plot(x,mem_utils[0],color='blue',linewidth=1,label='Canonical')
    p2, = plt.plot(x,mem_utils[1],color='red',linewidth=1,label='ProbSparse')
    p3, = plt.plot(x,mem_utils[2],color='green',linewidth=1,label='NeuralSparse')
    plt.xlabel("Time steps")
    plt.ylabel("memory")
    plt.legend([p1,p2,p3], ["Canonical","ProbSparse","NeuralSparse"], loc='upper left')
    plt.savefig('./performance/'+str(batch)+'/mem_comps.png')
    plt.close('all')
import os
batch_list = [1,8,16,32,64]
for batch in batch_list:
    if not os.path.exists(os.path.join('./performance',str(batch))):
        os.mkdir(os.path.join('./performance',str(batch)))
    performance(batch)
