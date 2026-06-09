import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate

import numpy as np
from scipy import signal

class BesselFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, fc):
        data = input_.detach().cpu().numpy()
        b, a = signal.bessel(4, fc, btype="low", analog=False) # order=4, cutoff=0.05 / order=2, cutoff=0.15
        y = signal.filtfilt(b, a, data)

        return torch.as_tensor(y.copy(), dtype=input_.dtype, device=input_.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        
        return grad_input

def bessel_filter(fc):
    def inner(x):
        return BesselFilter.apply(x, fc)

    return inner

class SNNModel3(nn.Module):
    def __init__(self, input_dim, layer1=32, layer2=48, output_dim=2,
                 batch_size=256, bin_window=0.2, num_steps=7, drop_rate=0.5,
                 beta=0.5, mem_thresh=0.5, spike_grad=surrogate.atan(alpha=2), fc=0.05):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate

        self.batch_size = batch_size
        self.bin_window_time = bin_window
        self.num_steps = num_steps
        self.sampling_rate = 0.004
        self.bin_window_size = int(self.bin_window_time / self.sampling_rate)
        self.step_size = self.bin_window_size // self.num_steps

        self.fc1 = nn.Linear(self.input_dim, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.output_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        self.norm_layer = nn.LayerNorm([self.num_steps, self.input_dim])

        self.beta = beta
        self.mem_thresh = mem_thresh
        self.spike_grad = spike_grad
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, threshold=self.mem_thresh, 
                              learn_beta=False, learn_threshold=False, init_hidden=True)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, threshold=self.mem_thresh, 
                              learn_beta=False, learn_threshold=False, init_hidden=True)
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, threshold=self.mem_thresh, 
                              learn_beta=False, learn_threshold=False, init_hidden=True, reset_mechanism="none")
        
        self.v_x = torch.nn.Parameter(torch.normal(0, 1, size=(1,), requires_grad=True))
        self.v_y = torch.nn.Parameter(torch.normal(0, 1, size=(1,), requires_grad=True))

        self.register_buffer("data_buffer", torch.zeros(1, input_dim).type(torch.float32), persistent=False)
        self.register_buffer("model_results_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.register_buffer("filter_results_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.register_buffer("filter_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.register_buffer("prediction_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.filter = bessel_filter(fc=fc)

    def reset_mem(self):
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()
        self.lif3.reset_hidden()

    def single_forward(self, x):
        x = self.norm_layer(x)
        self.reset_mem()
        input_ = torch.clone(x)

        for step in range(self.num_steps):
            cur1 = self.dropout(self.fc1(x[step, :]))
            spk1 = self.lif1(cur1)

            fc1_output = torch.clone(spk1)

            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)

            fc2_output = torch.clone(spk2)

            cur3 = self.fc3(spk2)
            spk3 = self.lif3(cur3)

        return self.lif3.mem.clone(), input_, fc1_output, fc2_output

    def forward(self, x):
        predictions = []
        block_filter_last, block_filter_middle, full_filtfilt = False, False, True
        win_size = 32
        first_run = False

        seq_length = x.shape[0]
        for seq in range(seq_length):
            current_seq = x[seq, :, :]
            self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)

            if self.data_buffer.shape[0] > self.bin_window_size:
                self.data_buffer = self.data_buffer[1:, :]
            else:
                first_run = True

            # Accumulate
            spikes = self.data_buffer.clone()
            
            acc_spikes = torch.zeros((self.num_steps, self.input_dim))
            for i in range(self.num_steps):
                temp = torch.sum(spikes[self.step_size*i:self.step_size*i+(self.step_size), :], dim=0)
                acc_spikes[i, :] = temp


            pred, input_, fc1_output, fc2_output = self.single_forward(acc_spikes)
            U_x = self.v_x*pred[0]
            U_y = self.v_y*pred[1]
            out = torch.stack((U_x, U_y), 0).permute(1, 0)

            if block_filter_last:

                self.model_results_buffer = torch.cat((self.model_results_buffer, out), dim=0)


                if self.filter_results_buffer.shape[0] <= win_size:
                    self.filter_results_buffer = torch.cat((self.filter_results_buffer, out), dim=0)
                else:
                    self.filter_results_buffer = self.filter_results_buffer[1:, :]
                    self.model_results_buffer = self.model_results_buffer[1:, :]

                    filtered_result = self.filter(self.model_results_buffer.t())
                    filtered_result = filtered_result.t()
                    self.filter_results_buffer = torch.cat((self.filter_results_buffer, filtered_result[-1, :].unsqueeze(0)), dim=0)
                    

                predictions.append(self.filter_results_buffer[-1, :].unsqueeze(0))

            elif  block_filter_middle:

                self.filter_buffer = torch.cat((self.filter_buffer, out), dim=0)
                self.prediction_buffer = torch.cat((self.prediction_buffer, out), dim=0)

                if self.filter_buffer.shape[0] > win_size:
                    self.filter_buffer = self.filter_buffer[1:, :]
                    self.prediction_buffer = self.prediction_buffer[1:, :]
                    
                if self.filter_buffer.shape[0] == win_size:
                    pred_ = self.filter_buffer.clone()
                    pred_ = pred_.t()
                    pred_ = self.filter(pred_)
                    pred_ = pred_.t()

                if self.filter_buffer.shape[0] < int(win_size//2):
                    predictions.append(self.prediction_buffer[-1, :].unsqueeze(dim=0))
                elif int(win_size//2) <= self.filter_buffer.shape[0] < win_size:
                    continue
                else:
                    predictions.append(pred_[int(pred_.shape[0]/2)-1, :].unsqueeze(dim=0))

            else:
                
                predictions.append(out)

        if block_filter_middle and first_run:
            for i in range(-int(win_size//2), 0):
                predictions.append(self.prediction_buffer[i, :].unsqueeze(dim=0))

        if full_filtfilt:
            prediction = torch.stack(predictions).squeeze(dim=1)
            if not self.training:
                filtered_result = self.filter(prediction.t())
                filtered_result = filtered_result.t()
                predictions = filtered_result
        else:
            predictions = torch.stack(predictions).squeeze(dim=1)


        return predictions