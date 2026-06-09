import torch
import torch.nn as nn

import numpy as np
from scipy import signal

class BesselFilter(torch.autograd.Function):
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

def bessel_filter():
    def inner(x):
        return BesselFilter.apply(x)

    return inner

## Define model ##
# The model defined here is a vanilla Fully Connected Network
class ANNModel2D(nn.Module):
    def __init__(self, input_dim, layer1=32, layer2=48, output_dim=2,
                 bin_window=0.2, sampling_rate=0.004, drop_rate=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate

        self.bin_window_time = bin_window
        self.sampling_rate = sampling_rate
        self.bin_window_size = int(self.bin_window_time / self.sampling_rate)

        self.fc1 = nn.Linear(self.input_dim, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.output_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        self.batchnorm1 = nn.BatchNorm1d(self.layer1)
        self.batchnorm2 = nn.BatchNorm1d(self.layer2)
        self.activation = nn.ReLU()

        self.register_buffer("data_buffer", torch.zeros(1, input_dim).type(torch.float32), persistent=False)
        self.register_buffer("model_results_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.register_buffer("filter_results_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.register_buffer("filter_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.register_buffer("prediction_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.filter = bessel_filter()
        self.run_num = 0

    def single_forward(self, x):
        input_ = torch.clone(x)

        x = self.activation(self.fc1(x.view(1, -1)))
        x = self.batchnorm1(x)

        fc1_output = torch.clone(x)

        x = self.activation(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)

        fc2_output = torch.clone(x)

        x = self.fc3(x)

        return x, input_, fc1_output, fc2_output

    def forward(self, x):
        block_filter_last, block_filter_middle, full_filtfilt, lfilt = False, False, False, False
        win_size = 32
        predictions = []
        first_run = False

        seq_length = x.shape[0]
        for seq in range(seq_length):
            current_seq = x[seq, :, :]
            self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)

            if self.data_buffer.shape[0] <= self.bin_window_size:
                predictions.append(torch.zeros(1, self.output_dim))
                first_run = True
            else:
                # Only pass input into model when the buffer size == bin_window_size
                if self.data_buffer.shape[0] > self.bin_window_size:
                    self.data_buffer = self.data_buffer[1:, :]

                # Accumulate
                spikes = self.data_buffer.clone()
                acc_spikes = torch.sum(spikes, dim=0)

                out, input_, fc1_output, fc2_output = self.single_forward(acc_spikes)

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
    
class ANNModel3D(nn.Module):
    def __init__(self, input_dim, layer1=32, layer2=48, output_dim=2,
                 bin_window=0.2, sampling_rate=0.004, num_steps=7, drop_rate=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate

        self.bin_window_time = bin_window
        self.num_steps = num_steps
        self.sampling_rate = sampling_rate
        self.bin_window_size = int(self.bin_window_time / self.sampling_rate)
        self.step_size = self.bin_window_size // self.num_steps

        self.fc1 = nn.Linear(self.input_dim*self.num_steps, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.output_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        self.batchnorm1 = nn.BatchNorm1d(self.layer1)
        self.batchnorm2 = nn.BatchNorm1d(self.layer2)
        self.activation = nn.ReLU()

        self.register_buffer("data_buffer", torch.zeros(1, input_dim).type(torch.float32), persistent=False)
        self.register_buffer("model_results_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.register_buffer("filter_results_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.register_buffer("filter_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.register_buffer("prediction_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
        self.filter = bessel_filter()
        # self.input_count = 0

    def single_forward(self, x):
        x = x.permute(1, 0).contiguous()
        # print("Post permute dim: ", x.shape)
        input_ = torch.clone(x)

        x = self.activation(self.fc1(x.view(1, -1)))
        x = self.batchnorm1(x)

        fc1_output = torch.clone(x)

        x = self.activation(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)

        fc2_output = torch.clone(x)

        x = self.fc3(x)

        return x, input_, fc1_output, fc2_output

    def forward(self, x):
        block_filter_last, block_filter_middle, full_filtfilt, lfilt = False, False, False, False
        win_size = 32
        predictions = []
        seq_length = x.shape[0]
        first_run = False
        run_num = 0

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


            # pred = self.single_forward(acc_spikes)
            out, input_, fc1_output, fc2_output = self.single_forward(acc_spikes)
            run_num += 1

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