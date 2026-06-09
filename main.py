import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset

from tqdm import tqdm
import numpy as np

import snntorch as snn
from snntorch import surrogate

from neurobench.benchmarks import Benchmark
from neurobench.datasets import PrimateReaching
from neurobench.models.torch_model import TorchModel
from neurobench.benchmarks.data_metrics import r2

from neurobench.examples.primate_reaching.ANN import ANNModel2D, ANNModel3D
from neurobench.examples.primate_reaching.SNN_3 import SNNModel3
from neurobench.examples.primate_reaching.LSTM import LSTMModel

from scipy import signal
import matplotlib.pyplot as plt
import random

DEVICE = "cuda"
BATCH_SIZE = 512

class MyDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, idx):
        sample = self.samples[:, idx, :]
        label = self.labels[:, idx]

        return sample, label
    
    def __len__(self):
        return self.samples.shape[1]

class MyDatasetLSTM(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, idx):
        sample = self.samples[:, idx]
        label = self.labels[:, idx]

        return sample, label
    
    def __len__(self):
        return self.samples.shape[1]

class ANNModel2DTraining(nn.Module):
    """
        A straightforward 3-layer fully-connected network for predicting x&y coordinate
        of the test subject.
        :param input_dim: input feature dimension
        :param layer1: Number of hidden neurons in the first layer
        :param layer2: Number of hidden neurons in the second layer
        :param output_dim: output feature dimension
        :param dropout_rate: Probability of dropout
        """

    def __init__(self, input_dim, layer1=32, layer2=48, output_dim=2, drop_rate=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate

        self.fc1 = nn.Linear(self.input_dim, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)
        self.batchnorm1 = nn.BatchNorm1d(layer1)
        self.batchnorm2 = nn.BatchNorm1d(layer2)

        self.batch_size = None

    def forward(self, x):
        self.batch_size = x.shape[0]
        x = torch.sum(x, dim=2)
        x = x.view(self.batch_size, -1)

        x = self.activation(self.fc1(x))
        x = self.batchnorm1(x)
        x = self.activation(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)
        x = self.fc3(x)

        return x

class ANNModel3DTraining(nn.Module):
    """
        A straightforward 3-layer fully-connected network for predicting x&y coordinate
        of the test subject.
        :param input_dim: input feature dimension
        :param layer1: Number of hidden neurons in the first layer
        :param layer2: Number of hidden neurons in the second layer
        :param output_dim: output feature dimension
        :param dropout_rate: Probability of dropout
        """

    def __init__(self, input_dim, layer1=32, layer2=48, output_dim=2, drop_rate=0.5, num_steps=7):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate
        self.num_steps = num_steps

        self.fc1 = nn.Linear(self.input_dim*self.num_steps, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)
        self.batchnorm1 = nn.BatchNorm1d(layer1)
        self.batchnorm2 = nn.BatchNorm1d(layer2)

        self.batch_size = None

    def forward(self, x):
        self.batch_size = x.shape[0]
        x = x.view(self.batch_size, -1)

        x = self.activation(self.fc1(x))
        x = self.batchnorm1(x)
        x = self.activation(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)
        x = self.fc3(x)

        return x

class SNNModelTraining(nn.Module):
    def __init__(self, input_dim, beta=0.5, mem_threshold=0.5, spike_grad2=surrogate.atan(alpha=2),
                 layer1=32, layer2=48, output_dim=2, dropout_rate=0.5, num_step=7):
        super().__init__()

        self.num_step = num_step
        self.input_dim = input_dim
        self.beta = beta
        self.spike_grad = spike_grad2
        self.mem_threshold = mem_threshold

        self.fc1 = nn.Linear(input_dim, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, output_dim)

        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, threshold=self.mem_threshold, learn_beta=True,
                              learn_threshold=True, init_hidden=True)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, threshold=self.mem_threshold, learn_beta=True,
                              learn_threshold=True, init_hidden=True)
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, threshold=self.mem_threshold, learn_beta=True,
                              learn_threshold=True, init_hidden=True, reset_mechanism="none")
        self.dropout = nn.Dropout(dropout_rate)

        self.v_x = torch.nn.Parameter(torch.normal(0, 1, size=(1,), requires_grad=True))
        self.v_y = torch.nn.Parameter(torch.normal(0, 1, size=(1,), requires_grad=True))
        self.norm_layer = nn.LayerNorm([self.num_step, self.input_dim])
        self.reset_mem = False

    def forward(self, x):
        if self.reset_mem:
            self.lif1.reset_hidden()
            self.lif2.reset_hidden()
            self.lif3.reset_hidden()

        x = self.norm_layer(x.permute(0, 2, 1))

        for step in range(self.num_step):
            input_ = x[:, step, :]
            cur1 = self.dropout(self.fc1(input_))
            spk1 = self.lif1(cur1)

            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)

            cur3 = self.fc3(spk2)
            spk3 = self.lif3(cur3)

        U_x = self.v_x*self.lif3.mem[:,0]
        U_y = self.v_y*self.lif3.mem[:,1]
        out = torch.stack((U_x, U_y), 1)

        return out   # Training Mode only need to return the output

class LSTMModelTraining(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, 
                 num_layers=1, droprate=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.norm_layer = nn.LayerNorm([self.input_dim], elementwise_affine=True)
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True)
        # self.lstm = nn.LSTMCell(self.input_dim, self.hidden_size)
        # The lstm module takes in input size of self.hidden_size
        # uses self.hidden_size hidden neurons
        # and contains only self.num_layers LSTM block
        self.dropout = nn.Dropout(droprate)
        self.fc1 = nn.Linear(self.hidden_size, self.output_dim) 

        self.init_lstm_weight()

    def init_lstm_weight(self):
        for m in self.modules():
            if type(m) in [torch.nn.LSTM]:
                for name, param in m.named_parameters():
                    if "bias" not in name:
                        torch.nn.init.xavier_normal_(param.data)

    def forward(self, x, h_t, c_t):
        """
            This block is for LSTM+LN+FILTERING
        """
        # Expected Input Feature Dimension: Batch_Size * Timesteps * Input_Features
        # x = self.norm_layer(x)

        # output, (h, c) = self.lstm(x, (h_t, c_t))
        # output = self.dropout(output)
        # x = self.fc1(output)

        # # This is with Filtering
        # x = x.squeeze().t()
        # filtered_result = self.filter(x)
        # filtered_result = filtered_result.t()

        # return filtered_result, (h, c)

        """
            This block is for LSTMCell+LN
        """
        # x = self.norm_layer(x)
        # output = []
        # for input_ in range(x.shape[1]):
        #     h_t, c_t = self.lstm(x[:, input_, :], (h_t, c_t))
        #     output.append(h_t)
        # output = torch.cat(output, dim=0)
        # output = self.dropout(output)
        # x = self.fc1(output)
        
        # return x, (h_t, c_t)

        """
            This block is for LSTM+LN
        """
        # Expected Input Feature Dimension: Batch_Size * Timesteps * Input_Features
        x = self.norm_layer(x)

        output, (h, c) = self.lstm(x, (h_t, c_t))
        output = self.dropout(output)

        x = self.fc1(output)
        x = x.squeeze(0)

        return x, (h, c)
    
        """
            This Block is for Pure LSTM Training
        """
        # if not self.training:
        #     input_ = x.clone().detach()
        # # Expected Input Feature Dimension: Batch_Size * Timesteps * Input_Features

        # output, (h, c) = self.lstm(x, (h_t, c_t))
        # output = self.dropout(output)

        # if not self.training:
        #     post_lstm = output.clone().detach()
        # x = self.fc1(output)
        # x = x.squeeze(0).t()

        # # This is without filtering
        # if self.training:
        #     return x, (h, c)
        # return x, (h, c), input_, post_lstm

def segment_bin_processing(samples, labels, stride=0.004, bin_width=0.032, model_type="LSTM"):
    advance_num = int(stride//0.004)
    bin_width_num = int(bin_width//0.004)

    new_samples, new_labels = [], []
    for sample, label in zip(samples, labels):
        # Sample Dim is (input_dim, seq_len), Label Dim is (2, seq_len)
        # print("hajjkfsa", sample.shape, label.shape)
        temp_sample = torch.zeros((sample.shape[0], int(sample.shape[1] // advance_num)), dtype=torch.float)
        temp_label = torch.zeros((label.shape[0], int(sample.shape[1] // advance_num)), dtype=torch.float)

        # Something similar to the original bin_width function
        for col in range(min(temp_sample.shape[1], temp_label.shape[1])):
            if col <  bin_width_num/advance_num:
                bin_end = int(col * advance_num)
                temp_sample[:, col] = torch.sum(sample[:, :bin_end], dim=1)
                # continue
            else:
                bin_start = int(col * advance_num - bin_width_num)
                bin_end = int(col * advance_num)
                temp_sample[:, col] = torch.sum(sample[:, bin_start:bin_end], dim=1)
            temp_label[:, col] = label[:, col*advance_num]

        # new_samples.append(temp_sample)
        if model_type == "ANN" or model_type == "LSTM":
            new_samples.append(temp_sample)
        else:
            new_samples.append((temp_sample > 0).float())
        new_labels.append(temp_label)

    return new_samples, new_labels

def reorder_dim(samples, labels):
    new_samples, new_labels = [], []

    for sample, label in zip(samples, labels):
        new_sample = torch.unsqueeze(sample.permute(1, 0), dim=0)
        new_samples.append(new_sample)
        new_labels.append(label.permute(1, 0))

    return new_samples, new_labels

def transform_to_3d(samples, labels, overlap=True, stride=0.004, bin_width=0.2, num_steps=7, model_type="ANN"):
    # print("Transformation Type is: ", model_type)
    # Determine if time window generated overlaps with one another
    if not overlap:
        advance_num = int(stride//0.004)
        bin_width_num = advance_num
    else:
        advance_num = int(stride//0.004)
        bin_width_num = int(bin_width//0.004)

    new_samples, new_labels = [], []
    for sample, label in zip(samples, labels):
        temp_sample = torch.zeros((sample.shape[0], int(sample.shape[1] // advance_num), bin_width_num), dtype=torch.float32)
        temp_label = torch.zeros((label.shape[0], int(sample.shape[1] // advance_num)), dtype=torch.float32)

        for col in range(temp_sample.shape[1]):
            if col <  bin_width_num/advance_num:
                bin_start = 0
                bin_end = int(col * advance_num)
                if col == 0:
                    bin_end = 1
                temp_sample[:, col, bin_start:bin_end] = sample[:, bin_start: bin_end]
                # continue
            else:
                bin_start = int(col * advance_num - bin_width_num)
                bin_end = int(col * advance_num)
                temp_sample[:, col, :] = sample[:, bin_start: bin_end]

            temp_label[:, col] = label[:, col * advance_num]

        if num_steps < bin_width_num:
            sum_num = bin_width_num // num_steps
            temp_sample_num_steps = torch.zeros((temp_sample.shape[0], temp_sample.shape[1], num_steps), dtype=torch.float32)
            for idx in range(num_steps):
                start_idx = idx*sum_num
                end_idx = idx*sum_num + sum_num
                temp_sample_num_steps[:, :, idx] = torch.sum(temp_sample[:, :, start_idx: end_idx], dim=2)

            if model_type == 'ANN':
                new_samples.append(temp_sample_num_steps)
            else:
                new_samples.append((temp_sample_num_steps > 0).float())
        else:

            if model_type == 'ANN':
                new_samples.append(temp_sample)
            else:
                new_samples.append((temp_sample > 0).float())

        new_labels.append(temp_label)
    
    return new_samples, new_labels

def training(dataset, net, model_weight_name):
    epochs = 50

    samples, labels = [], []
    for segment in dataset.time_segments:
        if segment[0]+1 in dataset.ind_train and (segment[1] - segment[0]) <= dataset.max_segment_length:
            samples.append(dataset.samples[:, segment[0]+1:segment[1]])
            labels.append(dataset.labels[:, segment[0]+1:segment[1]])

    samples = torch.cat(samples, dim=1)
    labels = torch.cat(labels, dim=1)
    transformed_samples, transformed_labels = transform_to_3d([samples], [labels], model_type="ANN")

    complete_samples = transformed_samples[0]
    complete_labels = transformed_labels[0]
    training_set = MyDataset(complete_samples, complete_labels)

    train_loader = DataLoader(
                            dataset=training_set,
                            batch_size=BATCH_SIZE,
                            drop_last=False,
                            shuffle=True,
                        )
    
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.AdamW(net.parameters(), lr=0.005, 
                                  betas=(0.9, 0.999), weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=64)
    best_training_r2, best_val_r2 = float("-inf"), float("-inf")

    for epoch in tqdm(range(epochs)):
        training_r2 = r2()
        net.train()
        # current_r2 = float("-inf")
        for i, (sample, label) in enumerate(train_loader):
            if "SNN" in MODEL_TYPE:
                net.reset_mem = True
            sample = sample.to(DEVICE)
            label = label.to(DEVICE)

            pred = net(sample)
            loss_val = criterion(pred, label)

            current_r2 = training_r2(net, pred.cpu(), (sample.cpu(), label.cpu()))
            if current_r2 > best_training_r2:
                best_training_r2 = current_r2

            optimiser.zero_grad()
            loss_val.backward()
            optimiser.step()
        print(f"{epoch} training R2 Score: {current_r2}")
        lr_scheduler.step()

        current_val_r2 = validation(dataset, net)
        if current_val_r2 > best_val_r2:
            best_val_r2 = current_val_r2
            torch.save(net.state_dict(), model_weight_name)
        print(f"{epoch} validation R2 Score: {current_val_r2}")
    
    print(f"Best Validation R2 is: {best_val_r2}")

def training_lstm(dataset, net, model_weight_name):
    epochs = 50

    samples, labels = [], []
    for segment in dataset.time_segments:
        if segment[0]+1 in dataset.ind_train and (segment[1] - segment[0]) <= dataset.max_segment_length:
        # if segment[0]+1 in dataset.ind_train:
            samples.append(dataset.samples[:, segment[0]+1:segment[1]])
            labels.append(dataset.labels[:, segment[0]+1:segment[1]])

    transformed_samples, transformed_labels = segment_bin_processing(samples, labels, model_type="LSTM")
    # Sample Dim is (input_dim, seq_len), Label Dim is (2, seq_len)
    transformed_samples, transformed_labels = reorder_dim(transformed_samples, transformed_labels)

    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, 
                                  betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=64)
    best_training_r2, best_val_r2 = float("-inf"), float("-inf")

    for epoch in tqdm(range(epochs)):
        training_r2 = r2()
        net.train()
        current_r2 = float("-inf")
        
        # Create shuffled ordering of our segments
        indices = [i for i in range(len(transformed_samples))]
        random.shuffle(indices) # This shuffles the indices in place

        for i in indices:
            sample = transformed_samples[i].to(DEVICE)
            label = transformed_labels[i].to(DEVICE)
            h_t = torch.zeros((1, 1, net.hidden_size)).to(DEVICE)
            c_t = torch.zeros((1, 1, net.hidden_size)).to(DEVICE)
            pred, (h, c) = net(sample, h_t, c_t)
            loss_val = criterion(pred, label)
            
            current_r2 = training_r2(net, pred.cpu(), (sample.cpu(), label.cpu()))
            if current_r2 > best_training_r2:
                best_training_r2 = current_r2

            optimiser.zero_grad()
            loss_val.backward()
            optimiser.step()

            h_t = h.detach()
            c_t = c.detach()
        print(f"{epoch} training R2 Score: {current_r2}")
        lr_scheduler.step()

        current_val_r2 = validation_lstm(dataset, net)
        if current_val_r2 > best_val_r2:
            best_val_r2 = current_val_r2
            torch.save(net.state_dict(), model_weight_name)
        print(f"{epoch} validation R2 Score: {current_val_r2}")
    
    print(f"Best Validation R2 is: {best_val_r2}")

def validation(dataset, net):
    net.eval()
    samples, labels = [], []
    for segment in dataset.time_segments:
        if segment[0]+1 in dataset.ind_val and (segment[1] - segment[0]) <= dataset.max_segment_length:
            samples.append(dataset.samples[:, segment[0]+1:segment[1]])
            labels.append(dataset.labels[:, segment[0]+1:segment[1]])

    transformed_samples, transformed_labels = transform_to_3d(samples, labels, model_type="ANN")

    complete_samples = torch.cat(transformed_samples, dim=1)
    complete_labels = torch.cat(transformed_labels, dim=1)

    val_set = MyDataset(complete_samples, complete_labels)
    
    val_loader = DataLoader(
                        dataset=val_set,
                        batch_size=BATCH_SIZE,
                        drop_last=False,
                        shuffle=False,
                    )
    
    val_r2 = r2()
    val_r2_final = float("-inf")
    with torch.no_grad():
        for i, (sample, label) in enumerate(val_loader):
            if "SNN" in MODEL_TYPE:
                net.reset_mem = True
            sample = sample.to(DEVICE)
            label = label.to(DEVICE)

            pred = net(sample)

            val_r2_final = val_r2(net, pred.cpu(), (sample.cpu(), label.cpu()))

    return val_r2_final

def validation_lstm(dataset, net):
    net.eval()
    samples, labels = [], []
    for segment in dataset.time_segments:
        if segment[0]+1 in dataset.ind_val and (segment[1] - segment[0]) <= dataset.max_segment_length:
            samples.append(dataset.samples[:, segment[0]+1:segment[1]])
            labels.append(dataset.labels[:, segment[0]+1:segment[1]])
    
    transformed_samples, transformed_labels = segment_bin_processing(samples, labels, model_type="LSTM")
    transformed_samples, transformed_labels = reorder_dim(transformed_samples, transformed_labels)

    val_r2 = r2()
    val_r2_final = float("-inf")
    with torch.no_grad():
        for i, (sample, label) in enumerate(zip(transformed_samples, transformed_labels)):
            sample = sample.to(DEVICE)
            label = label.to(DEVICE)

            h_t = torch.zeros((1, 1, net.hidden_size)).to(DEVICE)
            c_t = torch.zeros((1, 1, net.hidden_size)).to(DEVICE)
            pred, (h, c) = net(sample, h_t, c_t)

            val_r2_final = val_r2(net, pred.cpu(), (sample.squeeze(dim=0).cpu(), label.cpu()))

            h_t = h.detach()
            c_t = c.detach()

    return val_r2_final

def testing_no_segment_removal(dataset, net, model_weight_name):
    if MODEL_TYPE == "SNN" or MODEL_TYPE == "LSTM":
        test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=len(dataset.ind_test), shuffle=False)
    else:
        test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=BATCH_SIZE, shuffle=False)


    net.load_state_dict(torch.load(model_weight_name))
    net.eval()

    model = TorchModel(net)
    static_metrics = ["model_size"]
    data_metrics = ["r2"]
    # static_metrics = ["model_size", "connection_sparsity"]
    # data_metrics = ["r2", "activation_sparsity", "synaptic_operations"]
    benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics])
    results = benchmark.run()
    print("Final Result:")
    print(results)

def testing_segment_removal(dataset, net, model_weight_name):
    if MODEL_TYPE == "SNN" or MODEL_TYPE == "LSTM":
        test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=len(dataset.ind_test), shuffle=False)
    else:
        test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=BATCH_SIZE, shuffle=False)

    net.load_state_dict(torch.load(model_weight_name))
    net.eval()

    model = TorchModel(net)
    static_metrics = ["model_size"]
    data_metrics = ["r2"]
    # static_metrics = ["model_size", "connection_sparsity"]
    # data_metrics = ["r2", "activation_sparsity", "synaptic_operations"]
    benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics])
    results = benchmark.run()
    print("Final Result:")
    print(results)

def draw_raw_segments(dataset):
    labels = []
    break_points, loc = [], []
    offset = 0
    for i, segment in enumerate(dataset.time_segments):
        if i == 0:
            offset = segment[1]
            continue
        elif i < 6:
            labels.append(dataset.labels[:, segment[0]:segment[1]])
            break_points.append(dataset.labels[:, segment[1]].view(2, 1))
            loc.append(segment[1]-offset)
        else:
            break

    continuous_stream = torch.cat(labels, dim=1)
    break_points = torch.cat(break_points, dim=1)
    continuous_stream = continuous_stream.numpy()
    break_points = break_points.numpy()

    fig = plt.figure()
    plt.plot(continuous_stream[0, :], color="red", linewidth=1)
    plt.scatter(loc, break_points[0, :], marker="*", color="blue", linewidths=0.5)
    plt.ylim([-5, 5])
    plt.savefig("continuous_stream.jpg", dpi=1000)
    print("Figure Completed!")

if __name__ == "__main__":
    print("Start of Neurobench")

    filenames = ["indy_20160622_01", "indy_20160630_01", "indy_20170131_02", 
                 "loco_20170210_03", "loco_20170215_02", "loco_20170301_05"]

    SEGMENT_REMOVAL = False
    TRAIN_RATIO = 0.5
    SEED_VALUE = 1337
    MAX_SEGMENT_LENGTH = 2000
    torch.manual_seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    random.seed(SEED_VALUE)
    if torch.cuda.is_available():
        print("using cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(SEED_VALUE)

    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 0.11  # Normally, 0.1 works fine. Need to find the ideal value for lfilter (0.11 seems to be slightly better)
    DROPRATE = 0.5
    MODEL_TYPE = "ANN3D"  # Options: "ANN", "ANN3D", "SNN3", "LSTM"

    """
        Drawing Block
    """
    # fname = "indy_20160622_01"
    # dataset = PrimateReaching(file_path="./Dataset/", filename=fname,
    #                         num_steps=1, train_ratio=TRAIN_RATIO, bin_width=0.004,
    #                         biological_delay=0, max_segment_length=MAX_SEGMENT_LENGTH, remove_segments_inactive=False)
    # draw_raw_segments(dataset)

    for filename in filenames:
        print(f"Current File is: {filename}")
        current_filename = filename + ".mat"
        model_weight_name = filename + "_" + MODEL_TYPE + "_model_state_dict.pth"
        
        if "indy" in filename:
            input_dim = 96
        elif "loco" in filename:
            input_dim = 192

        dataset = PrimateReaching(file_path="./Dataset/", filename=current_filename,
                            num_steps=1, train_ratio=TRAIN_RATIO, bin_width=0.004,
                            biological_delay=0, max_segment_length=MAX_SEGMENT_LENGTH, remove_segments_inactive=False)

        # net = ANNModel2DTraining(input_dim=input_dim)
        net = ANNModel3DTraining(input_dim=input_dim)
        # net = SNNModelTraining(input_dim=input_dim)
        # net = LSTMModelTraining(input_dim=input_dim, hidden_size=32, output_dim=2, num_layers=1, droprate=DROPRATE)
        net.to(DEVICE)

        training(dataset, net, model_weight_name)
        # training_lstm(dataset, net, model_weight_name)

        # net = ANNModel2D(input_dim=input_dim)
        net = ANNModel3D(input_dim=input_dim)
        # net = SNNModel3(input_dim=input_dim)
        # net = LSTMModel(input_dim=input_dim, hidden_size=32, output_dim=2)
        if SEGMENT_REMOVAL:
            dataset = PrimateReaching(file_path="./Dataset/", filename=current_filename,
                            num_steps=1, train_ratio=TRAIN_RATIO, bin_width=0.004,
                            biological_delay=0, max_segment_length=MAX_SEGMENT_LENGTH, remove_segments_inactive=True)
            testing_segment_removal(dataset, net, model_weight_name)
        else:
            testing_no_segment_removal(dataset, net, model_weight_name)