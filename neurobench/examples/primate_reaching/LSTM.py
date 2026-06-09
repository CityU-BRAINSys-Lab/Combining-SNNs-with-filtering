import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate

import numpy as np
from scipy import signal

class BesselFilterLFilt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        data = input_.detach().cpu().numpy()
        b, a = signal.bessel(2, 0.5, btype="low", analog=False) # Best is 2, 0.5
        y = signal.lfilter(b, a, data)

        return torch.as_tensor(y.copy(), dtype=input_.dtype, device=input_.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        
        return grad_input

def bessel_filter_lfilt():
    def inner(x):
        return BesselFilterLFilt.apply(x)

    return inner

class BesselFilterFiltFilt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        data = input_.detach().cpu().numpy()
        b, a = signal.bessel(4, 0.05, btype="low", analog=False) # order=4, cutoff=0.05 / order=2, cutoff=0.15/ partial: order=2
        y = signal.filtfilt(b, a, data)
        # b, a = signal.bessel(1, 0.15, btype="low", analog=False)
        # y = signal.lfilter(b, a, data)

        return torch.as_tensor(y.copy(), dtype=input_.dtype, device=input_.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        
        return grad_input

def bessel_filter_filtfilt():
    def inner(x):
        return BesselFilterFiltFilt.apply(x)

    return inner

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim,
                 bin_width=0.032, droprate=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = 1
        self.sampling_rate = 0.004
        self.bin_width = bin_width
        self.bin_window_size = int(self.bin_width/self.sampling_rate)

        self.norm_layer = nn.LayerNorm([self.input_dim], elementwise_affine=True)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True)
        # self.lstm = nn.LSTMCell(self.input_dim, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.output_dim) 
        self.dropout = nn.Dropout(droprate)

        self.filter = bessel_filter_filtfilt()
        # self.filter = bessel_filter_lfilt()

        self.h_t = torch.zeros((1, self.hidden_size))
        self.c_t = torch.zeros((1, self.hidden_size))
        # self.h_t = torch.randn(1, self.hidden_size)
        # self.c_t = torch.randn(1, self.hidden_size)

        self.register_buffer("data_buffer", torch.zeros(1, input_dim).type(torch.float32), persistent=False)
        self.register_buffer("model_results_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.register_buffer("filter_results_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.register_buffer("filter_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.register_buffer("prediction_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        # self.reset_hidden_state()

    def reset_hidden_state_randn(self):
        self.h_t = torch.randn(1, self.hidden_size)
        self.c_t = torch.randn(1, self.hidden_size)

    def reset_hidden_state_zero(self):
        self.h_t = torch.zeros((1, self.hidden_size))
        self.c_t = torch.zeros((1, self.hidden_size))

    def single_forward(self, x):
        """
            LSTM Layer Version
        """
        x = self.norm_layer(x)
        output, (h, c) = self.lstm(x, (self.h_t, self.c_t))
        output = self.dropout(output)
        x = self.fc1(output)
        # x = x.squeeze(dim=0).t()  # Dunno if this is needed for Neurobench version

        self.h_t = h.detach()
        self.c_t = c.detach()

        return x
    
        """
            LSTMCell Version
        """
        # x = self.norm_layer(x)
        # h, c = self.lstm(x, (self.h_t, self.c_t))
        # x = self.fc1(h)

        # self.h_t = h
        # self.c_t = c

        # return x

    def single_forward_memory_access_calc(self, x):
        layernorm_sparsity = 1 - (torch.count_nonzero(x) / torch.numel(x))
        x = self.norm_layer(x)
        # normlayer_param = 0.0
        # for name, param in self.norm_layer.named_parameters():
        #     normlayer_param += torch.numel(param)
        # norm_layer_access = (1 - layernorm_sparsity)*normlayer_param

        lstm_sparsity = 1 - (torch.count_nonzero(x) / torch.numel(x))
        output, (h, c) = self.lstm(x, (self.h_t, self.c_t))
        # lstm_param = 0.0
        # for name, param in self.norm_layer.named_parameters():
        #     lstm_param += torch.numel(param)
        # lstm_access = (1-lstm_sparsity)*lstm_param

        output = self.dropout(output)
        fc_sparsity = 1 - (torch.count_nonzero(output) / torch.numel(output))
        x = self.fc1(output)
        # fc_param = 0.0
        # for name, param in self.fc1.named_parameters():
        #     fc_param += torch.numel(param)
        # fc_access = (1-fc_sparsity)*fc_param
        # x = x.squeeze(dim=0).t()  # Dunno if this is needed for Neurobench version

        # total_access = norm_layer_access + lstm_access + fc_access
        # print(f"Total Access count is: {total_access}")

        self.h_t = h.detach()
        self.c_t = c.detach()

        # return x, total_access
        return x, layernorm_sparsity, lstm_sparsity, fc_sparsity

    def forward(self, x):
        """
            This block is for complete filtering
        """
        # predictions = []
        # seq_length = x.shape[0]
        # # self.reset_hidden_state()
        # for seq in range(seq_length):
        #     current_seq = x[seq, :, :]
        #     self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)
        #     if self.data_buffer.shape[0] > self.bin_window_size:
        #         self.data_buffer = self.data_buffer[1:, :]

        #     # Accumulate
        #     spikes = self.data_buffer.clone()

        #     acc_spikes = torch.zeros((1, self.input_dim))
        #     temp = torch.sum(spikes[:, :], dim=0)
        #     # Sanity Check
        #     # print(temp.shape)   # This should be of the shape 1, 96 or 1, 192
        #     acc_spikes[0, :] = temp

        #     pred = self.single_forward(acc_spikes)
        #     predictions.append(pred)
        
        # predictions = torch.stack(predictions).squeeze(dim=1)

        # # Filter Block
        # predictions = predictions.t()
        # predictions = self.filter(predictions)
        # predictions = predictions.t()

        # return predictions

        """
            This block is for partial filtering with middle prediction
        """
        # predictions = []
        # seq_length = x.shape[0]
        # # self.reset_hidden_state()
        # for idx, seq in enumerate(range(seq_length)):
        #     current_seq = x[seq, :, :]
        #     self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)
        #     if self.data_buffer.shape[0] > self.bin_window_size:
        #         self.data_buffer = self.data_buffer[1:, :]

        #     # Accumulate
        #     spikes = self.data_buffer.clone()

        #     acc_spikes = torch.zeros((1, self.input_dim))
        #     temp = torch.sum(spikes[:, :], dim=0)
        #     # Sanity Check
        #     # print(temp.shape)   # This should be of the shape 1, 96 or 1, 192
        #     acc_spikes[0, :] = temp

        #     pred = self.single_forward(acc_spikes)
        #     self.filter_buffer = torch.cat((self.filter_buffer, pred), dim=0)
        #     self.prediction_buffer = torch.cat((self.prediction_buffer, pred), dim=0)
        #     if self.filter_buffer.shape[0] > 16:
        #         self.filter_buffer = self.filter_buffer[1:, :]
        #         self.prediction_buffer = self.prediction_buffer[1:, :]
        #     if self.filter_buffer.shape[0] == 16:
        #         pred_ = self.filter_buffer.clone()
        #         pred_ = pred_.t()
        #         pred_ = self.filter(pred_)
        #         pred_ = pred_.t()
        #     if idx < 8:
        #         predictions.append(self.prediction_buffer[-1, :])
        #     elif 8 <= idx < 16:
        #         continue
        #     else:
        #         predictions.append(pred_[int(pred_.shape[0]/2)-1, :])
        #         # predictions[-8] = pred_[int(pred_.shape[0]/2)-1, :]
        # for i in range(-8, 0):
        #     predictions.append(self.prediction_buffer[i, :])
        # predictions = torch.stack(predictions).squeeze(dim=1)
        # # print("akhjdlsfhak", predictions.shape)

        # return predictions
    
        """
            This block is for partial filtering with final prediction
        """
        # predictions = []
        # seq_length = x.shape[0]
        # # self.reset_hidden_state()
        # for idx, seq in enumerate(range(seq_length)):
        #     current_seq = x[seq, :, :]
        #     self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)
        #     if self.data_buffer.shape[0] > self.bin_window_size:
        #         self.data_buffer = self.data_buffer[1:, :]

        #     # Accumulate
        #     spikes = self.data_buffer.clone()

        #     acc_spikes = torch.zeros((1, self.input_dim))
        #     temp = torch.sum(spikes[:, :], dim=0)
        #     # Sanity Check
        #     # print(temp.shape)   # This should be of the shape 1, 96 or 1, 192
        #     acc_spikes[0, :] = temp

        #     pred = self.single_forward(acc_spikes)
        #     self.filter_buffer = torch.cat((self.filter_buffer, pred), dim=0)
        #     if self.filter_buffer.shape[0] > 16:
        #         self.filter_buffer = self.filter_buffer[1:, :]
        #     if self.filter_buffer.shape[0] == 16:
        #         pred = self.filter_buffer.clone()
        #         pred = pred.t()
        #         pred = self.filter(pred)
        #         pred = pred.t()
        #     predictions.append(pred[-1, :])

        # predictions = torch.stack(predictions).squeeze(dim=1)

        # return predictions

        """
            This block is for Memory Access Calculation
        """
        block_filter_last, block_filter_middle, full_filtfilt, lfilt = False, False, False, False
        win_size = 32
        predictions = []
        seq_length = x.shape[0]
        first_run = False

        for seq in range(seq_length):
            current_seq = x[seq, :, :]
            self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)
            if self.data_buffer.shape[0] > self.bin_window_size:
                self.data_buffer = self.data_buffer[1:, :]
            else:
                first_run = True

            # Accumulate
            spikes = self.data_buffer.clone()

            acc_spikes = torch.zeros((1, self.input_dim))
            temp = torch.sum(spikes[:, :], dim=0)
            # Sanity Check
            # print(temp.shape)   # This should be of the shape 1, 96 or 1, 192
            acc_spikes[0, :] = temp

            pred, ln_sparsity, lstm_sparsity, fc_sparsity = self.single_forward_memory_access_calc(acc_spikes)
            
            if block_filter_last:

                self.model_results_buffer = torch.cat((self.model_results_buffer, pred), dim=0)


                if self.filter_results_buffer.shape[0] <= win_size:
                    self.filter_results_buffer = torch.cat((self.filter_results_buffer, pred), dim=0)
                else:
                    self.filter_results_buffer = self.filter_results_buffer[1:, :]
                    self.model_results_buffer = self.model_results_buffer[1:, :]

                    filtered_result = self.filter(self.model_results_buffer.t())
                    filtered_result = filtered_result.t()
                    self.filter_results_buffer = torch.cat((self.filter_results_buffer, filtered_result[-1, :].unsqueeze(0)), dim=0)
                    

                predictions.append(self.filter_results_buffer[-1, :].unsqueeze(0))

            elif  block_filter_middle:

                self.filter_buffer = torch.cat((self.filter_buffer, pred), dim=0)
                self.prediction_buffer = torch.cat((self.prediction_buffer, pred), dim=0)

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
                
                predictions.append(pred)

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