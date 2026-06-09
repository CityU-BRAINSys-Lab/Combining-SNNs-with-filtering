# import torch
# import torch.nn as nn
# import snntorch as snn
# from snntorch import surrogate

# import numpy as np
# from scipy import signal

# class BesselFilter(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input_):
#         data = input_.detach().cpu().numpy()
#         b, a = signal.bessel(4, 0.05, btype="low", analog=False) # order=4, cutoff=0.05 / order=2, cutoff=0.15/ partial: order=2
#         y = signal.filtfilt(b, a, data)
#         # b, a = signal.bessel(1, 0.15, btype="low", analog=False)
#         # y = signal.lfilter(b, a, data)

#         return torch.as_tensor(y.copy(), dtype=input_.dtype, device=input_.device)

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
        
#         return grad_input

# def bessel_filter():
#     def inner(x):
#         return BesselFilter.apply(x)

#     return inner


# ## Define model ##
# class SNN2(nn.Module):

#     def __init__(self, window=50, input_size=96, hidden_size=50, tau=0.96, p=0.3, device='cpu'):
#         super().__init__()

#         # self.window = window
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = 2
#         self.surrogate = surrogate.fast_sigmoid(slope=20)

#         self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=False, device=device)
#         self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False, device=device)
#         self.fc_out = nn.Linear(self.hidden_size, self.output_size, bias=False, device=device)

#         self.lif1 = snn.Leaky(beta=tau, spike_grad=self.surrogate, threshold=1, learn_beta=False,
#                               learn_threshold=False, reset_mechanism='zero')
#         self.lif2 = snn.Leaky(beta=tau, spike_grad=self.surrogate, threshold=1, learn_beta=False,
#                               learn_threshold=False, reset_mechanism='zero')
#         self.lif_out = snn.Leaky(beta=tau, spike_grad=self.surrogate, threshold=1, learn_beta=False,
#                               learn_threshold=False, reset_mechanism='none')

#         self.dropout = nn.Dropout(p)
#         self.mem1, self.mem2, self.mem3 = None, None, None

#         # self.register_buffer('inp', torch.zeros(window, self.input_size))
#         self.register_buffer("data_buffer", torch.zeros(1, input_size).type(torch.float32), persistent=False)
#         self.register_buffer("model_results_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
#         self.register_buffer("filter_results_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
#         self.register_buffer("filter_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
#         self.register_buffer("prediction_buffer", torch.zeros(1, 2).type(torch.float32), persistent=False)
#         self.filter = bessel_filter()

#     def reset(self):
#         self.mem1 = self.lif1.init_leaky()
#         self.mem2 = self.lif2.init_leaky()
#         self.mem3 = self.lif_out.init_leaky()

#     def single_forward(self, x):
#         x = x.squeeze() # convert shape (1, input_dim) to (input_dim)
#         cur1 = self.dropout(self.fc1(x))
#         spk1, self.mem1 = self.lif1(cur1, self.mem1)

#         cur2 = self.dropout(self.fc2(spk1))
#         spk2, self.mem2 = self.lif2(cur2, self.mem2)

#         cur3 = self.fc_out(spk2)
#         _, self.mem3 = self.lif_out(cur3, self.mem3)

#         return self.mem3.clone()

#     def forward(self, x):
#         # here x is expected to be shape (len_series, 1, input_dim)
#         predictions = []
#         block_filter_last, block_filter_middle, full_filtfilt, lfilt = False, False, False, False
#         win_size = 32
#         if self.training:
#             x = torch.sum(x, dim=2)
#         self.reset()

#         for sample in range(x.shape[0]):
#             # predictions.append(self.single_forward(x[sample, ...]))
#             # x_ = x[:,:,sample]
#             pred = self.single_forward(x[sample, ...])
#             # pred = self.single_forward(x_)
#             pred = pred.unsqueeze(0)
#             if block_filter_last:

#                 self.model_results_buffer = torch.cat((self.model_results_buffer, pred), dim=0)


#                 if self.filter_results_buffer.shape[0] <= win_size:
#                     self.filter_results_buffer = torch.cat((self.filter_results_buffer, pred), dim=0)
#                 else:
#                     self.filter_results_buffer = self.filter_results_buffer[1:, :]
#                     self.model_results_buffer = self.model_results_buffer[1:, :]

#                     filtered_result = self.filter(self.model_results_buffer.t())
#                     filtered_result = filtered_result.t()
#                     self.filter_results_buffer = torch.cat((self.filter_results_buffer, filtered_result[-1, :].unsqueeze(0)), dim=0)
#                     # self.filter_results_buffer = torch.cat((self.filter_results_buffer, filtered_result[7, :].unsqueeze(0)), dim=0)
                    

#                 predictions.append(self.filter_results_buffer[-1, :].unsqueeze(0))

#             elif  block_filter_middle:

#                 self.filter_buffer = torch.cat((self.filter_buffer, pred), dim=0)
#                 self.prediction_buffer = torch.cat((self.prediction_buffer, pred), dim=0)

#                 if self.filter_buffer.shape[0] > win_size:
#                     self.filter_buffer = self.filter_buffer[1:, :]
#                     self.prediction_buffer = self.prediction_buffer[1:, :]

#                 if self.filter_buffer.shape[0] == win_size:
#                     pred_ = self.filter_buffer.clone()
#                     pred_ = pred_.t()
#                     pred_ = self.filter(pred_)
#                     pred_ = pred_.t()

#                 if self.filter_buffer.shape[0] < win_size/2:
#                     predictions.append(self.prediction_buffer[-1, :].unsqueeze(dim=0))
#                 elif win_size/2 <= self.filter_buffer.shape[0] < win_size:
#                     continue
#                 else:
#                     predictions.append(pred_[int(pred_.shape[0]/2)-1, :].unsqueeze(dim=0))

#             else:
#                 predictions.append(pred)

                
#         if full_filtfilt or lfilt:
#             predictions = torch.stack(predictions).squeeze(dim=1)
#             if not self.training and predictions.shape[0] >= win_size:
#                 filtered_result = self.filter(predictions.t())
#                 filtered_result = filtered_result.t()
#                 predictions = filtered_result

#         elif block_filter_middle:
#             for i in range(-int(win_size/2), 0):
#                 predictions.append(self.prediction_buffer[i, :].unsqueeze(dim=0))
        
#         predictions = torch.stack(predictions).squeeze(dim=1)

#         # predictions = torch.stack(predictions)
#         return predictions

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

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
class SNN2(nn.Module):

    def __init__(self, window=50, input_size=96, hidden_size=50, tau=0.96, p=0.3, device='cpu'):
        super().__init__()

        # self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2
        self.surrogate = surrogate.fast_sigmoid(slope=20)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=False, device=device)
        self.fc_out = nn.Linear(self.hidden_size, self.output_size, bias=False, device=device)

        self.lif1 = snn.Leaky(beta=tau, spike_grad=self.surrogate, threshold=1, learn_beta=False,
                              learn_threshold=False, reset_mechanism='zero')
        self.lif_out = snn.Leaky(beta=tau, spike_grad=self.surrogate, threshold=1, learn_beta=False,
                              learn_threshold=False, reset_mechanism='none')

        self.dropout = nn.Dropout(p)
        self.mem1, self.mem2 = None, None

        # self.register_buffer('inp', torch.zeros(window, self.input_size))

    def reset(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif_out.init_leaky()

    def single_forward(self, x):
        x = x.squeeze() # convert shape (1, input_dim) to (input_dim)
        cur1 = self.dropout(self.fc1(x))
        spk1, self.mem1 = self.lif1(cur1, self.mem1)

        cur2 = self.fc_out(spk1)
        _, self.mem2 = self.lif_out(cur2, self.mem2)

        return self.mem2.clone()

    def forward(self, x):
        # here x is expected to be shape (len_series, 1, input_dim)
        predictions = []
        block_filter_last, block_filter_middle, full_filtfilt, lfilt = False, False, False, False
        win_size = 32
        self.reset()

        for sample in range(x.shape[0]):
            # predictions.append(self.single_forward(x[sample, ...]))
            # x_ = x[:,:,sample]
            pred = self.single_forward(x[sample, ...])
            # pred = self.single_forward(x_)
            pred = pred.unsqueeze(0)
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
                    # self.filter_results_buffer = torch.cat((self.filter_results_buffer, filtered_result[7, :].unsqueeze(0)), dim=0)
                    

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

                if self.filter_buffer.shape[0] < win_size/2:
                    predictions.append(self.prediction_buffer[-1, :].unsqueeze(dim=0))
                elif win_size/2 <= self.filter_buffer.shape[0] < win_size:
                    continue
                else:
                    predictions.append(pred_[int(pred_.shape[0]/2)-1, :].unsqueeze(dim=0))

            else:
                predictions.append(pred)

                
        if full_filtfilt or lfilt:
            predictions = torch.stack(predictions).squeeze(dim=1)
            if not self.training and predictions.shape[0] >= win_size:
                filtered_result = self.filter(predictions.t())
                filtered_result = filtered_result.t()
                predictions = filtered_result

        elif block_filter_middle:
            for i in range(-int(win_size/2), 0):
                predictions.append(self.prediction_buffer[i, :].unsqueeze(dim=0))
        
        predictions = torch.stack(predictions).squeeze(dim=1)

        # predictions = torch.stack(predictions)
        return predictions