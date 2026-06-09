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
from neurobench.models.snntorch_models import SNNTorchModel

from neurobench.benchmarks.data_metrics import r2

from neurobench.examples.primate_reaching.ANN import ANNModel2D, ANNModel3D
from neurobench.examples.primate_reaching.SNN_3 import SNNModel3
from neurobench.examples.primate_reaching.SNN2 import SNN2
from neurobench.examples.primate_reaching.SNN_Streaming import SNNModelStreaming

from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.model_selection import KFold

import time
import random
from scipy import signal 

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

    
class SNNModelStreamingTraining(nn.Module):
    def __init__(self, input_dim, beta=0.5, mem_threshold=0.5, spike_grad2=surrogate.atan(alpha=2),
                 layer1=32, layer2=48, output_dim=2, dropout_rate=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.beta = beta
        self.spike_grad = spike_grad2
        self.mem_threshold = mem_threshold

        self.fc1 = nn.Linear(input_dim, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, output_dim)

        self.lif1 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad, threshold=0.4, learn_beta=True,
                              learn_threshold=True, init_hidden=False)
        self.lif2 = snn.Leaky(beta=0.7, spike_grad=self.spike_grad, threshold=0.6, learn_beta=True,
                              learn_threshold=True, init_hidden=False, reset_mechanism="none")
        self.lif3 = snn.Leaky(beta=0.8, spike_grad=self.spike_grad, threshold=0.5, learn_beta=True,
                              learn_threshold=True, init_hidden=False, reset_mechanism="none")
        self.dropout = nn.Dropout(dropout_rate)

        self.v_x = torch.nn.Parameter(torch.normal(0, 1, size=(1,), requires_grad=True))
        self.v_y = torch.nn.Parameter(torch.normal(0, 1, size=(1,), requires_grad=True))

        self.reset_mem = False

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        mem3_rec = []

        seq_len = x.shape[1]
        for step in range(seq_len):
            input_ = x[:, step]
            cur1 = self.dropout(self.fc1(input_))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.dropout(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            mem3_rec.append(mem3)

        temp = torch.stack(mem3_rec, dim=0)
        U_x = self.v_x*temp[:, 0]
        U_y = self.v_y*temp[:, 1]
        predictions = torch.stack((U_x, U_y), 1)

        return predictions  # Training Mode only need to return the output

def segment_bin_processing(samples, labels, overlap=True, stride=0.004, bin_width=0.004, num_steps=7, model_type="ANN"):
        advance_num = int(stride//0.004)
        bin_width_num = int(bin_width//0.004)

        new_samples, new_labels = [], []
        for i, (sample, label) in enumerate(zip(samples, labels)):
            sample = sample.t()
            temp_sample = torch.zeros((int(sample.shape[0] // advance_num), sample.shape[1]), dtype=torch.float32)
            temp_label = torch.zeros((label.shape[0], int(sample.shape[0] // advance_num)), dtype=torch.float32)

            # Something similar to the original bin_width function
            for col in range(min(temp_sample.shape[0], temp_label.shape[1])):
                if col <  bin_width_num/advance_num:
                    continue
                else:
                    bin_start = int(col * advance_num - bin_width_num)
                    bin_end = int(col * advance_num)
                    temp_sample[col, :] = torch.sum(sample[bin_start:bin_end, :], dim=0)
                temp_label[:, col] = label[:, col*advance_num]

            if model_type == "ANN" or model_type == "LSTM":
                new_samples.append(temp_sample.t())
            else:
                new_samples.append((temp_sample.t() > 0).float())
            new_labels.append(temp_label)

        samples = new_samples
        labels = new_labels
        return new_samples, new_labels

def training(dataset, net, model_weight_name):
    epochs = 50
    samples, labels = [], []

    for segment in dataset.time_segments:
        if segment[0]+1 in dataset.ind_train and (segment[1] - segment[0]) < dataset.max_segment_length:
            samples.append(dataset.samples[:, segment[0]+1:segment[1]])
            labels.append(dataset.labels[:, segment[0]+1:segment[1]])
    transformed_samples, transformed_labels = segment_bin_processing(samples, labels, model_type=MODEL_TYPE)

    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.AdamW(net.parameters(), lr=0.005, 
                                  betas=(0.9, 0.999), weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=50)
    best_training_r2, best_val_r2 = float("-inf"), float("-inf")

    for epoch in tqdm(range(epochs)):
        training_r2 = r2()
        net.train()

        ## Shuffle
        indices = [i for i in range(len(transformed_samples))]
        random.shuffle(indices)

        for i in indices:
            sample = transformed_samples[i]
            label = transformed_labels[i]
            if MODEL_TYPE == "SNN":
                net.reset_mem = True
            sample = sample.to(DEVICE)
            label = label.to(DEVICE)
            label = label.t()

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
    
def validation(dataset, net):
    net.eval()
    samples, labels = [], []
    for segment in dataset.time_segments:
        if segment[0]+1 in dataset.ind_val and (segment[1] - segment[0]) < dataset.max_segment_length:
            samples.append(dataset.samples[:, segment[0]+1:segment[1]])
            labels.append(dataset.labels[:, segment[0]+1:segment[1]])

    transformed_samples, transformed_labels = segment_bin_processing(samples, labels, model_type=MODEL_TYPE)
    
    val_r2 = r2()
    val_r2_final = float("-inf")
    with torch.no_grad():
        for i, (sample, label) in enumerate(zip(transformed_samples, transformed_labels)):
            if MODEL_TYPE == "SNN":
                net.reset_mem = True
            sample = sample.to(DEVICE)
            label = label.to(DEVICE)
            label = label.t()

            pred = net(sample)

            val_r2_final = val_r2(net, pred.cpu(), (sample.cpu(), label.cpu()))
    return val_r2_final

def testing(dataset, net, model_weight_name):
    if MODEL_TYPE == "SNN":
        test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=len(dataset.ind_test), shuffle=False)
    else:
        test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=BATCH_SIZE, shuffle=False)



    net.load_state_dict(torch.load(model_weight_name))
    net.eval()

    static_metrics = ["model_size", "connection_sparsity"]
    data_metrics = ["r2", "activation_sparsity", "synaptic_operations"]

    model = SNNTorchModel(net)

    benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics])
    results = benchmark.run()

    # print("Final Result:")
    # # print(results)
    # print("Footprint: {}".format(results['model_size']))
    # print("Connection sparsity: {}".format(results['connection_sparsity']))
    # print("Activation sparsity: {}".format(results['activation_sparsity']))
    # print("Dense: {}".format(results['synaptic_operations']['Dense']))
    # print("MACs: {}".format(results['synaptic_operations']['Effective_MACs']))
    # print("ACs: {}".format(results['synaptic_operations']['Effective_ACs']))
    print("R2: {}".format(results['r2']))     
    return results['r2']

def kfold(dataset, k=5, current_round=0):

    kfold = KFold(n_splits=k, shuffle=False)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        if current_round == fold:
            dataset.ind_train = train_ids
            dataset.ind_val = test_ids[:int(len(test_ids)//2)]
            dataset.ind_test = test_ids[int(len(test_ids)//2):]

if __name__ == "__main__":
    print("Start of Neurobench")

    filenames = ["indy_20160622_01", "indy_20160630_01", "indy_20170131_02", 
                 "loco_20170210_03", "loco_20170215_02", "loco_20170301_05"]

    MODEL_TYPE = "SNN"

    torch.manual_seed(678910)
    np.random.seed(1234)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(678910)

    fc_choice_list = [0.0403, 0.0527, 0.04839, 0.05259, 0.05053, 0.05052]

    for file_id, filename in enumerate(filenames):
        print(f"Current File is: {filename}")
        current_filename = filename + ".mat"
        model_weight_name = "./" + filename + "_model_state_dict.pth"
        
        if "indy" in filename:
            input_dim = 96
        elif "loco" in filename:
            input_dim = 192


        dataset = PrimateReaching(file_path="./Dataset/", filename=current_filename,
                            num_steps=1, train_ratio=0.5, bin_width=0.004,
                            biological_delay=0, max_segment_length=2000, remove_segments_inactive=False) 
        
        # kfold(dataset, k=5, current_round=k) ### Using K-Fold Cross Validation, comment out if not needed

        net = SNNModelStreamingTraining(input_dim=input_dim)
        net.to(DEVICE)

        training(dataset, net, model_weight_name)

        net = SNNModelStreaming(input_dim=input_dim)
        r2_num = testing(dataset, net, model_weight_name)