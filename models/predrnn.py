import numpy as np
import torch, cv2
import torch.nn as nn
import math 
from models.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell, ConvLSTMCell
        
def reverse_schedule_sampling_exp(args, numiterations, imgwd, imght, imgchannels):
    # setting the r_eta 
    if numiterations < args.r_sampling_step1: r_eta = 0.5
    elif numiterations < args.r_sampling_step2:
        r_eta = 1.0 - 0.5 * math.exp(-float(numiterations- args.r_sampling_step1) / args.r_exp_alpha)
    else:   r_eta = 1.0

    # setting the eta
    if numiterations < args.r_sampling_step1:    eta = 0.5
    elif numiterations < args.r_sampling_step2:
        eta = 0.5 - (0.5 / (args.r_sampling_step2 - args.r_sampling_step1)) * (numiterations - args.r_sampling_step1)
    else:   eta = 0.0

    r_random_flip = np.random.random_sample((args.batch_size, args.input_len - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample((args.batch_size, args.seq_len - args.input_len - 1))
    true_token = (random_flip < eta)

    ones = np.ones((imgwd , imght, imgchannels))
    zeros = np.zeros((imgwd , imght, imgchannels))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.seq_len - 2):
            if j < args.input_len - 1:
                if r_true_token[i, j]:  real_input_flag.append(ones)
                else:   real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.input_len - 1)]: real_input_flag.append(ones)
                else:   real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,(args.batch_size,args.seq_len - 2 , imgwd,
                                imght, imgchannels))

    return real_input_flag

def schedule_sampling(args, eta, numiterations, imgwd, imght, imgchannels):

    zeros = np.zeros((args.batch_size, args.seq_len - args.input_len - 1,
                    imgwd , imght ,  imgchannels))

    if not args.scheduled_sampling:
        return 0.0, zeros

    if numiterations < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate # linear decay (eta_k in the paper)
    else:   eta = 0.0

    random_flip = np.random.random_sample((args.batch_size, args.seq_len - args.input_len - 1))
    true_token = (random_flip < eta)

    ones = np.ones((imgwd , imght, imgchannels))
    zeros = np.zeros((imgwd , imght, imgchannels))

    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.seq_len - args.input_len - 1):
            if true_token[i, j]:    real_input_flag.append(ones)
            else:   real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,(args.batch_size, args.seq_len - args.input_len -1, imgwd,
                                imght, imgchannels))
    
    return eta, real_input_flag

def test_mask(args, imgwd, imght, imgchannels):
    if args.reverse_scheduled_sampling == 1:    mask_input = 1
    else:   mask_input = args.input_len

    real_input_flag = np.zeros((args.batch_size, args.seq_len-mask_input-1, imgwd, imght, imgchannels))

    if args.reverse_scheduled_sampling == 1:
        real_input_flag[:, :args.input_len - 1, :, :] = 1.0

    return real_input_flag

class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.args = args 
        self.num_hidden = args.num_hidden 
        self.num_layers = len(self.num_hidden) 
        cell_list = [] 

        for i in range(self.num_layers):
            if i==0:    in_channel = args.img_channels 
            else:   in_channel = self.args.num_hidden[0]
            num_hidden = self.args.num_hidden[0]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden, self.args.filter_size, self.args.stride)
            )

        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers-1],args.img_channels, kernel_size=1, stride=self.args.stride, padding=0, bias=False)
    
    def forward(self, seq_tensors, numiterations):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        eta = self.args.sampling_start_value
        _, imgchannels, imght, imgwd = seq_tensors.shape
        seq_tensors = seq_tensors.contiguous().view(self.args.batch_size, self.args.seq_len, imgchannels, imght, imgwd)

        if self.args.is_training:
            if self.args.reverse_scheduled_sampling == 1:
                mask_true = reverse_schedule_sampling_exp(self.args, numiterations, imgwd, imght, imgchannels)
            else:   eta, mask_true = schedule_sampling(self.args, eta, numiterations, imgwd, imght, imgchannels)
        else:   
            mask_true = test_mask(self.args, imgwd, imght, imgchannels)

        frames = seq_tensors 
        mask_true = torch.from_numpy(mask_true).contiguous().permute(0, 1, 4, 2, 3)
        mask_true = mask_true.to(self.args.device).float()
        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([self.args.batch_size, self.num_hidden[i], imght, imgwd]).to(self.args.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([self.args.batch_size, self.num_hidden[0], imght, imgwd]).to(self.args.device)

        for t in range(self.args.seq_len -1):
            if self.args.reverse_scheduled_sampling:
                if t==0:    x_t = (frames[:, t])
                else:     x_t = (mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen)

            else:
                if t < self.args.input_len:  x_t = (frames[:, t])
                else:
                    x_t = (mask_true[:, t - self.args.input_len] * frames[:, t] + \
                          (1 - mask_true[:, t - self.args.input_len]) * x_gen)

            h_t[0], c_t[0], memory = self.cell_list[0](x_t, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        return next_frames


class Standard_RNN(nn.Module):
    def __init__(self, args):
        super(Standard_RNN, self).__init__()
        self.args = args 
        self.num_hidden = args.num_hidden 
        self.num_layers = len(self.num_hidden) 
        cell_list = [] 

        for i in range(self.num_layers):
            if i==0:    in_channel = args.img_channels 
            else:   in_channel = self.args.num_hidden[0]
            num_hidden = self.args.num_hidden[0]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden, kernel_size=self.args.filter_size)
            )

        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers-1],args.img_channels, kernel_size=1, stride=self.args.stride, padding=0, bias=False)
    
    def forward(self, seq_tensors, numiterations):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        eta = self.args.sampling_start_value
        _, imgchannels, imght, imgwd = seq_tensors.shape
        seq_tensors = seq_tensors.contiguous().view(self.args.batch_size, self.args.seq_len, imgchannels, imght, imgwd)

        if self.args.is_training:
            if self.args.reverse_scheduled_sampling == 1:
                mask_true = reverse_schedule_sampling_exp(self.args, numiterations, imgwd, imght, imgchannels)
            else:   eta, mask_true = schedule_sampling(self.args, eta, numiterations, imgwd, imght, imgchannels)
        else:   
            mask_true = test_mask(self.args, imgwd, imght, imgchannels)

        frames = seq_tensors 
        mask_true = torch.from_numpy(mask_true).contiguous().permute(0, 1, 4, 2, 3)
        mask_true = mask_true.to(self.args.device).float()
        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([self.args.batch_size, self.num_hidden[i], imght, imgwd]).to(self.args.device)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.args.seq_len -1):
            if self.args.reverse_scheduled_sampling:
                if t==0:    x_t = (frames[:, t])
                else:     x_t = (mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen)

            else:
                if t < self.args.input_len:  x_t = (frames[:, t])
                else:
                    x_t = (mask_true[:, t - self.args.input_len] * frames[:, t] + \
                          (1 - mask_true[:, t - self.args.input_len]) * x_gen)

            h_t[0], c_t[0] = self.cell_list[0](x_t, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        return next_frames