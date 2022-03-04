from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import GPUtil
import psutil
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import torch.utils.tensorboard as tf
import warnings
import matplotlib.pyplot as plt
import global_var
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'DM':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        nParams = sum([p.nelement() for p in self.model.parameters()])
        with open(folder_path+'results.txt','a') as f:
            f.write('\nPara:'+str(nParams)+".Setting:"+str(global_var.get_value())+'.\n')
            f.write(setting+'\n')
            f.write('train_time,corr,mae,mse,gpu_used,gpu_util,cpu_used,cpu_util,para\n')
        time_now = time.time()
        writer = tf.SummaryWriter('./logs')
        tf_steps = 0
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):            
                iter_count += 1
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                writer.add_scalar('loss',loss,tf_steps)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                tf_steps += 1
                writer.add_scalar('lr',model_optim.param_groups[0]['lr'],tf_steps)

            train_time = time.time()-epoch_time
            print("Epoch: {} cost time: {}".format(epoch+1, train_time))
            torch.save(self.model.state_dict(), path+'/'+str(epoch)+'_epoch.pth')
        
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # scheduler.step()
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            self.test(setting,train_time,nParams)
            adjust_learning_rate(model_optim, epoch+1, self.args)
        return self.model

    def test(self, setting, timing,nParams):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        preds = []
        trues = []
        mem = psutil.virtual_memory()
        gpu = GPUtil.getGPUs()[0]
        GPUtil.showUtilization()
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        rse, corr, mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('corr:{:.3f},mae:{:.3f},mse:{:.3f}'.format(corr, mae, mse))
        with open(folder_path+'results.txt','a') as f:
            f.write('{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(timing, round(corr,3), 
            round(mae,3), round(mse,3), round(gpu.memoryUsed/1024,3), round(gpu.memoryUtil*100,3),round(mem.used/1024/1024,3),round(mem.percent,3),round(nParams/1000000,3)))
        return

    def continue_test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        pre_length = 1
        preds_list = [12,48,96,144,192,240,288]
        label_len =  self.args.label_len
        for steps in preds_list:
            preds = []
            trues = []
            preds_all = None
            trues_all =None
            start = None
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if i == (steps+1)*pre_length:
                    break
                if i % pre_length == 0: 
                    if i == 0:
                        start = batch_x.cuda()
                        pred, _ = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                        preds.append(pred.squeeze())
                    else:
                        start = torch.cat([start[:,pre_length:,:],pred],dim=1)
                        label_y =  torch.zeros(batch_y.shape)
                        label_y[:,-pre_length:,:] = 0
                        label_y[:,:label_len,:] = start[:,-label_len:,:]
                        pred, _ = self._process_one_batch(test_data, start, label_y, batch_x_mark, batch_y_mark)
                        preds.append(pred.squeeze())
                trues.append(batch_y[:,-1,:])

            for i in range(len(preds)):
                if i == 0:
                    preds_all = preds[i].unsqueeze(0)
                else:
                    preds_all = torch.cat([preds_all,preds[i].unsqueeze(0)],dim=0) 
            for i in range(len(trues)):
                if i == 0:
                    trues_all = trues[i]
                else:
                    trues_all = torch.cat([trues_all,trues[i]],dim=0) 
            trues_all = trues_all.detach().cpu().numpy()
            preds_all = preds_all.detach().cpu().numpy()
            print(trues_all.shape,preds_all.shape)
            # result save
            folder_path = './results/' + 'continues' +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            logs = open(os.path.join(folder_path,'logs.txt'),'a')
            rse, corr, mae, mse, rmse, mape, mspe = metric(preds_all, trues_all)
            print('steps:{},rse:{},corr:{},mae:{},mse:{},rmse:{}.'.format(steps,rse, corr, mae, mse, rmse))
            logs.write('steps:{},rse:{},corr:{},mae:{},mse:{},rmse:{}.\n'.format(steps,rse, corr, mae, mse, rmse))
            
            plt_fold = folder_path+str(steps)+'/'
            if not os.path.exists(plt_fold):
                    os.mkdir(plt_fold)
            x = np.linspace(0,preds_all.shape[0],preds_all.shape[0])
            columns =['SP1A_DASD_RESP','SP1A_DASD_RATE','SP1B_DASD_RESP','SP1B_DASD_RATE','SP1C_DASD_RESP',
                'SP1C_DASD_RATE','SP1D_DASD_RESP','SP1D_DASD_RATE','SP1A_MEM','SP1B_MEM','SP1C_MEM','SP1D_MEM','N_TASKS','TPS','SP1A_THOUT','SP1B_THOUT','SP1C_THOUT','SP1D_THOUT','SYSPLEX_MIPS','RESP_TIME']
            for i in range(preds_all.shape[1]):
                p2, = plt.plot(x,(preds_all[:,i]),color='red',linewidth=1,label='Predict')
                p1, = plt.plot(x,(trues_all[:,i]),color='blue',linewidth=1,label='GT')
                plt.xlabel("mins/time")
                plt.ylabel(columns[i])
                plt.legend([p2,p1], ["Predict","GT"], loc='upper left')
                plt.savefig(plt_fold+columns[i]+'_'+'.png')
                plt.close('all')
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        self.model.eval()
        preds = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            while True:
                pass
            preds.append(pred.detach().cpu().numpy())
            print(batch_x_mark[0][0:5])
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        time_now = time.time()
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y
