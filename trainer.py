import os
import numpy as np
import pickle

import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from model import LeNetShiftTeacher, ResNetShiftTeacher
from biggan import Discriminator as BigShiftTeacher


class Trainer(object):
    def __init__(self, dataset, val, params):
        ### Misc ###
        self.p = params
        self.device = params.device

        ### Make Dirs ###
        self.log_dir = params.log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        ### load/save params
        if params.load_params:
            with open(os.path.join(params.log_dir, 'params.pkl'), 'rb') as file:
                params = pickle.load(file)
        else:
            with open(os.path.join(params.log_dir,'params.pkl'), 'wb') as file:
                pickle.dump(params, file)

        ### Make Models ###
        if self.p.res_net:
            self.model = ResNetShiftTeacher(self.p).to(self.device)
        elif self.p.biggan:
            self.model = BigShiftTeacher(self.p).to(self.device)
        else:
            self.model = LeNetShiftTeacher(self.p).to(self.device)
        
        if self.p.ngpu>1:
            self.model = nn.DataParallel(self.model)

        self.opt = optim.Adam(self.model.parameters(), lr=self.p.lr)
        
        self.grad_scaler = GradScaler()

        ### Make Data Generator ###
        self.generator_train = DataLoader(dataset, batch_size=self.p.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.generator_val = DataLoader(val, batch_size=val.__len__(), shuffle=True, num_workers=4)
        self.gen = self.inf_train_gen()

        ### Prep Training
        self.bce = nn.BCELoss()
        self.losses = []
        self.val_losses = []

    def inf_train_gen(self):
        while True:
            for data in self.generator_train:
                yield data

    def start_from_checkpoint(self):
        step = 0
        files = [f for f in os.listdir(self.log_dir)]
        if len(files) < 3:
            checkpoint = os.path.join(self.log_dir, 'checkpoint.pt')
        else:
            files.remove('checkpoint.pt')
            files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            checkpoint = os.path.join(self.log_dir, files[-1])

        if os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint)
            step = state_dict['step']
            self.model.load_state_dict(state_dict['model'])
            self.opt.load_state_dict(state_dict['opt'])
            self.losses = state_dict['loss']
            self.val_losses = state_dict['val_loss']
            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, step):
        if step < self.p.niters - 1001:
            name = 'checkpoint.pt'
        else:
            name = f'checkpoint_{step}.pt'

        torch.save({
        'step': step,
        'model': self.model.state_dict(),
        'opt': self.opt.state_dict(),
        'loss': self.losses,
        'val_loss': self.val_losses,
        }, os.path.join(self.log_dir, name))

    def log(self, step):
        if step % self.p.steps_per_log == 0:
            shift, cl = self.losses[-1]
            val_shift, val_cl = self.val_losses[-1]
            print('[%d/%d] Shift Loss: %.2f, Class Loss: %.2f, Val Shift: %.2f, Val Class %.2f'
                        % (step, self.p.niters, shift, cl, val_shift, val_cl))

        if step % self.p.steps_per_checkpoint == 0:
            self.save_checkpoint(step)

    def log_final(self, step):
        shift, cl = self.losses[-1]
        val_shift, val_cl = self.val_losses[-1]
        print('[%d/%d] Shift Loss: %.2f, Class Loss: %.2f, Val Shift: %.2f, Val Class %.2f'
                    % (step, self.p.niters, shift, cl, val_shift, val_cl))
        self.save_checkpoint(step)

    def step(self):
        for p in self.model.parameters():
                    p.requires_grad = True
            
        data, shifts, y = next(self.gen)
        data, shifts, y = data.to(self.p.device), shifts.to(self.p.device), y.to(self.p.device)
        self.model.zero_grad()
        with autocast():
            shift_pred, y_pred = self.model(data)

            shifts, shift_pred = shifts[y == 1], shift_pred[y == 1]
            shift_loss = torch.log(torch.mean(torch.abs(shift_pred - shifts)))
            cl_loss = self.bce(y_pred, y)
            loss = shift_loss + cl_loss
        
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.opt)
        self.grad_scaler.update()

        for p in self.model.parameters():
            p.requires_grad = False

        return shift_loss.item(), cl_loss

    def val_step(self):
        with torch.no_grad():
            data, shifts, y = next(iter(self.generator_val))
            data, shifts, y = data.to(self.p.device), shifts.to(self.p.device), y.to(self.p.device)
            with autocast():
                shift_pred, y_pred = self.model(data)

                shifts, shift_pred = shifts[y == 1], shift_pred[y == 1]
                shift_loss = torch.log(torch.mean(torch.abs(shift_pred - shifts)))
                cl_loss = self.bce(y_pred, y)

        return shift_loss.item(), cl_loss.item()

    def train(self):
        step_done = self.start_from_checkpoint()

        print("Starting Training...")
        for i in range(step_done, self.p.niters):
            
            loss = self.step()
            val_loss = self.val_step()

            self.losses.append(loss)
            self.val_losses.append(val_loss)

            self.log(i)
                
        self.log_final(i)

        print('...Done')
