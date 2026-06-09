import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import math
import numpy as np



#################### block=32, order=4 -----------------------------------------------------------------------------------------------------------------------------------------------------


"""
    For Remove_segment=False
"""
#define aesthetics for plot
ANN_2D_MARKER = '.'
ANN_3D_MARKER = 'v'
LSTM_MARKER = 's'
SNN_FLAT_MARKER = 'p'
SNN_STREAMING_MARKER = '*'
PREVIOUS_WORK_MARKER = 'd'

BASE_COLOUR = '#4B61A8'
LFILT_COLOUR = 'crimson'
FILTFILT_COLOUR = 'lime'
BLOCK_FILT_COLOUR = '#FDB76D'
LSTM_21_COLOR = '#7fab87'
LSTM_35_COLOR = '#c89a9c'
SNN2_COLOR = '#7fab87'

line_size = 2
marker_size = 250

# Standard Model
ANN_2D_ACC = [0.5818]
ANN_3D_ACC = [0.6013]
LSTM_ACC = [0.6508333333]
SNN_FLAT_ACC = [0.6218666667]
SNN_STREAMING_ACC = [0.6111833333]
ANN_2D_COMPUTE = [4969.761667*3]
ANN_3D_COMPUTE = [11507.07333*3]
LSTM_COMPUTE = [22687.85*3]
SNN_FLAT_COMPUTE = [32256*3]
SNN_STREAMING_COMPUTE = [971.2609833*2]
ANN_2D_MEMORY = [27160/1024]
ANN_3D_MEMORY = [137752/1024]
LSTM_MEMORY = [93128/1024]
SNN_FLAT_MEMORY = [33996/1024]
SNN_STREAMING_MEMORY = [25932/1024]
ANN_2D_BLOCK_FILT_ACC = [0.6165]
ANN_2D_BLOCK_FILT_COMPUTE = [4974.76*3]
ANN_2D_BLOCK_FILT_MEMORY = [26.5429]
ANN_3D_BLOCK_FILT_ACC = [0.6646]
ANN_3D_BLOCK_FILT_COMPUTE = [11512.07333*3]
ANN_3D_BLOCK_FILT_MEMORY = [134.5429]
LSTM_BLOCK_FILT_ACC = [0.6711]
LSTM_BLOCK_FILT_COMPUTE = [22692.97*3]
LSTM_BLOCK_FILT_MEMORY = [93143/1024]
SNN_FLAT_BLOCK_FILT_ACC = [0.6729]
SNN_FLAT_BLOCK_FILT_COMPUTE = [32261*3]
SNN_FLAT_BLOCK_FILT_MEMORY = [33996/1024]
SNN_STREAMING_BLOCK_FILT_ACC = [0.6449]
SNN_STREAMING_BLOCK_FILT_COMPUTE = [976.2609833*2]
SNN_STREAMING_BLOCK_FILT_MEMORY = [25932/1024]

# Small Model
ANN_2D_ACC_SMALL = [0.57444]
ANN_3D_ACC_SMALL = [0.5846833333]
SNN_FLAT_ACC_SMALL = [0.6172333333]
SNN_STREAMING_ACC_SMALL = [0.60515]
ANN_2D_COMPUTE_SMALL = [2345.2008*3]
ANN_3D_COMPUTE_SMALL = [5712.325*3]
SNN_FLAT_COMPUTE_SMALL = [16128*3]
SNN_STREAMING_COMPUTE_SMALL = [491.6788333*2]
ANN_2D_MEMORY_SMALL = [13080/1024]
ANN_3D_MEMORY_SMALL = [136752/1024]
SNN_FLAT_MEMORY_SMALL = [20428/1024]
SNN_STREAMING_MEMORY_SMALL = [13580/1024]
ANN_2D_BLOCK_FILT_ACC_SMALL = [0.6017666667]
ANN_3D_BLOCK_FILT_ACC_SMALL = [0.62355]
SNN_FLAT_BLOCK_FILT_ACC_SMALL = [0.6614939333]
SNN_FLAT_BLOCK_FILT_COMPUTE_SMALL = [16133*3]
SNN_FLAT_BLOCK_FILT_MEMORY_SMALL = [20428/1024]
SNN_STREAMING_BLOCK_FILT_ACC_SMALL = [0.6383702667]
SNN_STREAMING_BLOCK_FILT_COMPUTE_SMALL = [496.6788333*2]
SNN_STREAMING_BLOCK_FILT_MEMORY_SMALL = [13580/1024]

LSTM_21_acc = [0.6109]
LSTM_21_COMPUTE = [1154*3]
LSTM_21_MEMORY = [5000]
LSTM_35_acc = [0.6746]
LSTM_35_COMPUTE = [872393.04*3]
LSTM_35_MEMORY = [3417.35]
SNN2_acc = [0.6292]
SNN2_COMPUTE = [202*2]
SNN2_MEMORY = [30]


COMPUTE_PARETO_FRONTIER = np.array([[LSTM_35_acc, LSTM_35_COMPUTE], [SNN_FLAT_BLOCK_FILT_ACC, SNN_FLAT_BLOCK_FILT_COMPUTE], [LSTM_BLOCK_FILT_ACC, LSTM_BLOCK_FILT_COMPUTE], [ANN_3D_BLOCK_FILT_ACC, ANN_3D_BLOCK_FILT_COMPUTE],
                           [SNN_STREAMING_BLOCK_FILT_ACC, SNN_STREAMING_BLOCK_FILT_COMPUTE], [SNN_STREAMING_BLOCK_FILT_ACC_SMALL, SNN_STREAMING_BLOCK_FILT_COMPUTE_SMALL], [SNN2_acc, SNN2_COMPUTE]])
MEMORY_PARETO_FRONTIER = np.array([[LSTM_35_acc, LSTM_35_MEMORY], [SNN_FLAT_BLOCK_FILT_ACC, SNN_FLAT_BLOCK_FILT_MEMORY], [SNN_FLAT_BLOCK_FILT_ACC_SMALL, SNN_FLAT_BLOCK_FILT_MEMORY_SMALL], 
                          [SNN_STREAMING_BLOCK_FILT_ACC_SMALL, SNN_STREAMING_BLOCK_FILT_MEMORY_SMALL]])

font_choice = {"fontname": "Times New Roman"}

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111)
artist1 = plt.scatter(ANN_2D_ACC, ANN_2D_COMPUTE, color=BASE_COLOUR, marker=ANN_2D_MARKER, s=marker_size)
plt.scatter(ANN_2D_BLOCK_FILT_ACC, ANN_2D_BLOCK_FILT_COMPUTE, color=BLOCK_FILT_COLOUR, marker=ANN_2D_MARKER, s=marker_size)
plt.scatter(ANN_2D_ACC_SMALL, ANN_2D_COMPUTE_SMALL, color=BASE_COLOUR, marker=ANN_2D_MARKER, edgecolors="black", s=marker_size)

artist2 = plt.scatter(ANN_3D_ACC, ANN_3D_COMPUTE, color=BASE_COLOUR, marker=ANN_3D_MARKER, s=marker_size)
plt.scatter(ANN_3D_BLOCK_FILT_ACC, ANN_3D_BLOCK_FILT_COMPUTE, color=BLOCK_FILT_COLOUR, marker=ANN_3D_MARKER, s=marker_size)
plt.scatter(ANN_3D_ACC_SMALL, ANN_3D_COMPUTE_SMALL, color=BASE_COLOUR, marker=ANN_3D_MARKER, edgecolors="black", s=marker_size)

artist3 = plt.scatter(LSTM_ACC, LSTM_COMPUTE, color=BASE_COLOUR, marker=LSTM_MARKER, s=marker_size)
plt.scatter(LSTM_BLOCK_FILT_ACC, LSTM_BLOCK_FILT_COMPUTE, color=BLOCK_FILT_COLOUR, marker=LSTM_MARKER, s=marker_size)

artist4 = plt.scatter(SNN_FLAT_ACC, SNN_FLAT_COMPUTE, color=BASE_COLOUR, marker=SNN_FLAT_MARKER, s=marker_size)
plt.scatter(SNN_FLAT_BLOCK_FILT_ACC, SNN_FLAT_BLOCK_FILT_COMPUTE, color=BLOCK_FILT_COLOUR, marker=SNN_FLAT_MARKER, s=marker_size)
plt.scatter(SNN_FLAT_ACC_SMALL, SNN_FLAT_COMPUTE_SMALL, color=BASE_COLOUR, marker=SNN_FLAT_MARKER, edgecolors="black", s=marker_size)
plt.scatter(SNN_FLAT_BLOCK_FILT_ACC_SMALL, SNN_FLAT_BLOCK_FILT_COMPUTE_SMALL, color=BLOCK_FILT_COLOUR, marker=SNN_FLAT_MARKER, edgecolors="black", s=marker_size)

artist5 = plt.scatter(SNN_STREAMING_ACC, SNN_STREAMING_COMPUTE, color=BASE_COLOUR, marker=SNN_STREAMING_MARKER, s=marker_size)
plt.scatter(SNN_STREAMING_BLOCK_FILT_ACC, SNN_STREAMING_BLOCK_FILT_COMPUTE, color=BLOCK_FILT_COLOUR, marker=SNN_STREAMING_MARKER, s=marker_size)
plt.scatter(SNN_STREAMING_ACC_SMALL, SNN_STREAMING_COMPUTE_SMALL, color=BASE_COLOUR, marker=SNN_STREAMING_MARKER, edgecolors="black", s=marker_size)
plt.scatter(SNN_STREAMING_BLOCK_FILT_ACC_SMALL, SNN_STREAMING_BLOCK_FILT_COMPUTE_SMALL, color=BLOCK_FILT_COLOUR, marker=SNN_STREAMING_MARKER, edgecolors="black", s=marker_size)

artist6 = plt.scatter(LSTM_21_acc, LSTM_21_COMPUTE, color=LSTM_21_COLOR, marker=LSTM_MARKER, s=marker_size)
artist7 = plt.scatter(LSTM_35_acc, LSTM_35_COMPUTE, color=LSTM_35_COLOR, marker=LSTM_MARKER, s=marker_size)
artist8 = plt.scatter(SNN2_acc, SNN2_COMPUTE, color=SNN2_COLOR, marker=SNN_STREAMING_MARKER, s=marker_size)

plt.plot(COMPUTE_PARETO_FRONTIER[:, 0], COMPUTE_PARETO_FRONTIER[:, 1], color="black")

handles = [artist1, artist2, artist3, artist4, artist5, artist6, artist7, artist8]
labels = ['ANN', 'ANN_3D', 'LSTM', 'SNN_3D', 'SNN_STREAMING', 'LSTM[130]', 'LSTM[187]', 'SNN2[130]']

compute_increase_SNN_FLAT = np.array([[SNN_FLAT_ACC, SNN_FLAT_COMPUTE], [SNN_FLAT_BLOCK_FILT_ACC, SNN_FLAT_BLOCK_FILT_COMPUTE]])
compute_increase_small_SNN_FLAT = np.array([[SNN_FLAT_ACC_SMALL, SNN_FLAT_COMPUTE_SMALL], [SNN_FLAT_BLOCK_FILT_ACC_SMALL, SNN_FLAT_BLOCK_FILT_COMPUTE_SMALL]])
compute_increase_SNN_STREAMING = np.array([[SNN_STREAMING_ACC, SNN_STREAMING_COMPUTE], [SNN_STREAMING_BLOCK_FILT_ACC, SNN_STREAMING_BLOCK_FILT_COMPUTE]])
compute_increase_small_SNN_STREAMING = np.array([[SNN_STREAMING_ACC_SMALL, SNN_STREAMING_COMPUTE_SMALL], [SNN_STREAMING_BLOCK_FILT_ACC_SMALL, SNN_STREAMING_BLOCK_FILT_COMPUTE_SMALL]])           
compute_increase_LSTM = np.array([[LSTM_ACC, LSTM_COMPUTE], [LSTM_BLOCK_FILT_ACC, LSTM_BLOCK_FILT_COMPUTE]])                  
compute_increase_ANN = np.array([[ANN_2D_ACC, ANN_2D_COMPUTE], [ANN_2D_BLOCK_FILT_ACC, ANN_2D_BLOCK_FILT_COMPUTE]])          
compute_increase_ANN_3D = np.array([[ANN_3D_ACC, ANN_3D_COMPUTE], [ANN_3D_BLOCK_FILT_ACC, ANN_3D_BLOCK_FILT_COMPUTE]])  

plt.plot(compute_increase_SNN_FLAT[:, 0], compute_increase_SNN_FLAT[:, 1], ':', color='black', linewidth=1.5)
plt.plot(compute_increase_small_SNN_FLAT[:, 0], compute_increase_small_SNN_FLAT[:, 1], ':', color='black', linewidth=1.5)
plt.plot(compute_increase_SNN_STREAMING[:, 0], compute_increase_SNN_STREAMING[:, 1], ':', color='black', linewidth=1.5)
plt.plot(compute_increase_small_SNN_STREAMING[:, 0], compute_increase_small_SNN_STREAMING[:, 1], ':', color='black', linewidth=1.5)
plt.plot(compute_increase_LSTM[:, 0], compute_increase_LSTM[:, 1], ':', color='black', linewidth=1.5)
plt.plot(compute_increase_ANN[:, 0], compute_increase_ANN[:, 1], ':', color='black', linewidth=1.5)
plt.plot(compute_increase_ANN_3D[:, 0], compute_increase_ANN_3D[:, 1], ':', color='black', linewidth=1.5)

vec = 0.002
plt.plot(compute_increase_SNN_FLAT[1, 0] - vec, compute_increase_SNN_FLAT[1, 1], '>', color='black', linewidth=2.5)
plt.plot(compute_increase_small_SNN_FLAT[1, 0] - vec, compute_increase_small_SNN_FLAT[1, 1], '>', color='black', linewidth=2.5)
plt.plot(compute_increase_SNN_STREAMING[1, 0] - vec, compute_increase_SNN_STREAMING[1, 1], '>', color='black', linewidth=2.5)
plt.plot(compute_increase_small_SNN_STREAMING[1, 0] - vec, compute_increase_small_SNN_STREAMING[1, 1], '>', color='black', linewidth=2.5)
plt.plot(compute_increase_LSTM[1, 0] - vec, compute_increase_LSTM[1, 1], '>', color='black', linewidth=2.5)
plt.plot(compute_increase_ANN[1, 0] - vec, compute_increase_ANN[1, 1], '>', color='black', linewidth=2.5)
plt.plot(compute_increase_ANN_3D[1, 0] - vec, compute_increase_ANN_3D[1, 1], '>', color='black', linewidth=2.5)

plt.yscale("log")
plt.xlabel('Accuracy ($R^{2}_{50}$)', fontsize=20, **font_choice, fontweight='bold')
plt.ylabel('Ops/Inference', fontsize=20, **font_choice, fontweight='bold')
plt.xticks(fontsize=16, **font_choice, fontweight='bold')
plt.yticks(fontsize=16, **font_choice, fontweight='bold')
plt.tick_params(axis="both", length=8, width=2, which="major")
plt.tick_params(axis="both", length=4, width=2, which="minor")
plt.legend(handles=handles, labels=labels, loc="best", framealpha=0.5, prop = {'size':13})
ax.spines.top.set_linewidth(3)
ax.spines.bottom.set_linewidth(3)
ax.spines.left.set_linewidth(3)
ax.spines.right.set_linewidth(3)
ax.set_rasterized(True)
plt.savefig("./Compute_vs_Accuracy_50_Split_SNN_ANN.png", bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111)
artist1 = plt.scatter(ANN_2D_ACC, ANN_2D_MEMORY, color=BASE_COLOUR, marker=ANN_2D_MARKER, s=marker_size)
plt.scatter(ANN_2D_BLOCK_FILT_ACC, ANN_2D_BLOCK_FILT_MEMORY, color=BLOCK_FILT_COLOUR, marker=ANN_2D_MARKER, s=marker_size)
plt.scatter(ANN_2D_ACC_SMALL, ANN_2D_MEMORY_SMALL, color=BASE_COLOUR, marker=ANN_2D_MARKER, edgecolors="black", s=marker_size)

artist2 = plt.scatter(ANN_3D_ACC, ANN_3D_MEMORY, color=BASE_COLOUR, marker=ANN_3D_MARKER, s=marker_size)
plt.scatter(ANN_3D_BLOCK_FILT_ACC, ANN_3D_BLOCK_FILT_MEMORY, color=BLOCK_FILT_COLOUR, marker=ANN_3D_MARKER, s=marker_size)
plt.scatter(ANN_3D_ACC_SMALL, ANN_3D_MEMORY_SMALL, color=BASE_COLOUR, marker=ANN_3D_MARKER, edgecolors="black", s=marker_size)

artist3 = plt.scatter(LSTM_ACC, LSTM_MEMORY, color=BASE_COLOUR, marker=LSTM_MARKER, s=marker_size)
plt.scatter(LSTM_BLOCK_FILT_ACC, LSTM_BLOCK_FILT_MEMORY, color=BLOCK_FILT_COLOUR, marker=LSTM_MARKER, s=marker_size)

artist4 = plt.scatter(SNN_FLAT_ACC, SNN_FLAT_MEMORY, color=BASE_COLOUR, marker=SNN_FLAT_MARKER, s=marker_size)
plt.scatter(SNN_FLAT_BLOCK_FILT_ACC, SNN_FLAT_BLOCK_FILT_MEMORY, color=BLOCK_FILT_COLOUR, marker=SNN_FLAT_MARKER, s=marker_size)
plt.scatter(SNN_FLAT_ACC_SMALL, SNN_FLAT_MEMORY_SMALL, color=BASE_COLOUR, marker=SNN_FLAT_MARKER, edgecolors="black", s=marker_size)
plt.scatter(SNN_FLAT_BLOCK_FILT_ACC_SMALL, SNN_FLAT_BLOCK_FILT_MEMORY_SMALL, color=BLOCK_FILT_COLOUR, marker=SNN_FLAT_MARKER, edgecolors="black", s=marker_size)

artist5 = plt.scatter(SNN_STREAMING_ACC, SNN_STREAMING_MEMORY, color=BASE_COLOUR, marker=SNN_STREAMING_MARKER, s=marker_size)
plt.scatter(SNN_STREAMING_BLOCK_FILT_ACC, SNN_STREAMING_BLOCK_FILT_MEMORY, color=BLOCK_FILT_COLOUR, marker=SNN_STREAMING_MARKER, s=marker_size)
plt.scatter(SNN_STREAMING_ACC_SMALL, SNN_STREAMING_MEMORY_SMALL, color=BASE_COLOUR, marker=SNN_STREAMING_MARKER, edgecolors="black", s=marker_size)
plt.scatter(SNN_STREAMING_BLOCK_FILT_ACC_SMALL, SNN_STREAMING_BLOCK_FILT_MEMORY_SMALL, color=BLOCK_FILT_COLOUR, marker=SNN_STREAMING_MARKER, edgecolors="black", s=marker_size)

artist6 = plt.scatter(LSTM_21_acc, LSTM_21_MEMORY, color=LSTM_21_COLOR, marker=LSTM_MARKER, s=marker_size)
artist7 = plt.scatter(LSTM_35_acc, LSTM_35_MEMORY, color=LSTM_35_COLOR, marker=LSTM_MARKER, s=marker_size)
artist8 = plt.scatter(SNN2_acc, SNN2_MEMORY, color=SNN2_COLOR, marker=SNN_STREAMING_MARKER, s=marker_size)


plt.plot(MEMORY_PARETO_FRONTIER[:, 0], MEMORY_PARETO_FRONTIER[:, 1], color="black")

handles = [artist1, artist2, artist3, artist4, artist5, artist6, artist7, artist8]
labels = ['ANN', 'ANN_3D', 'LSTM', 'SNN_3D', 'SNN_STREAMING', 'LSTM[130]', 'LSTM[187]', 'SNN2[130]']

mem_increase_SNN_FLAT = np.array([[SNN_FLAT_ACC, SNN_FLAT_MEMORY], [SNN_FLAT_BLOCK_FILT_ACC, SNN_FLAT_BLOCK_FILT_MEMORY]])
mem_increase_small_SNN_FLAT = np.array([[SNN_FLAT_ACC_SMALL, SNN_FLAT_MEMORY_SMALL], [SNN_FLAT_BLOCK_FILT_ACC_SMALL, SNN_FLAT_BLOCK_FILT_MEMORY_SMALL]])   
mem_increase_SNN_STREAMING = np.array([[SNN_STREAMING_ACC, SNN_STREAMING_MEMORY], [SNN_STREAMING_BLOCK_FILT_ACC, SNN_STREAMING_BLOCK_FILT_MEMORY]])
mem_increase_small_SNN_STREAMING = np.array([[SNN_STREAMING_ACC_SMALL, SNN_STREAMING_MEMORY_SMALL], [SNN_STREAMING_BLOCK_FILT_ACC_SMALL, SNN_STREAMING_BLOCK_FILT_MEMORY_SMALL]])  
mem_increase_LSTM = np.array([[LSTM_ACC, LSTM_MEMORY], [LSTM_BLOCK_FILT_ACC, LSTM_BLOCK_FILT_MEMORY]])           
mem_increase_ANN = np.array([[ANN_2D_ACC, ANN_2D_MEMORY], [ANN_2D_BLOCK_FILT_ACC, ANN_2D_BLOCK_FILT_MEMORY]])   
mem_increase_ANN_3D = np.array([[ANN_3D_ACC, ANN_3D_MEMORY], [ANN_3D_BLOCK_FILT_ACC, ANN_3D_BLOCK_FILT_MEMORY]])  


plt.plot(mem_increase_SNN_FLAT[:, 0], mem_increase_SNN_FLAT[:, 1], ':', color='black', linewidth=1.5)
plt.plot(mem_increase_small_SNN_FLAT[:, 0], mem_increase_small_SNN_FLAT[:, 1], ':', color='black', linewidth=1.5)
plt.plot(mem_increase_SNN_STREAMING[:, 0], mem_increase_SNN_STREAMING[:, 1], ':', color='black', linewidth=1.5)
plt.plot(mem_increase_small_SNN_STREAMING[:, 0], mem_increase_small_SNN_STREAMING[:, 1], ':', color='black', linewidth=1.5)
plt.plot(mem_increase_LSTM[:, 0], mem_increase_LSTM[:, 1], ':', color='black', linewidth=1.5)
plt.plot(mem_increase_ANN[:, 0], mem_increase_ANN[:, 1], ':', color='black', linewidth=1.5)
plt.plot(mem_increase_ANN_3D[:, 0], mem_increase_ANN_3D[:, 1], ':', color='black', linewidth=1.5)

vec = 0.002
plt.plot(mem_increase_SNN_FLAT[1, 0] - vec, mem_increase_SNN_FLAT[1, 1], '>', color='black', linewidth=2.5)
plt.plot(mem_increase_small_SNN_FLAT[1, 0] - vec, mem_increase_small_SNN_FLAT[1, 1], '>', color='black', linewidth=2.5)
plt.plot(mem_increase_SNN_STREAMING[1, 0] - vec, mem_increase_SNN_STREAMING[1, 1], '>', color='black', linewidth=2.5)
plt.plot(mem_increase_small_SNN_STREAMING[1, 0] - vec, mem_increase_small_SNN_STREAMING[1, 1], '>', color='black', linewidth=2.5)
plt.plot(mem_increase_LSTM[1, 0] - vec, mem_increase_LSTM[1, 1], '>', color='black', linewidth=2.5)
plt.plot(mem_increase_ANN[1, 0] - vec, mem_increase_ANN[1, 1], '>', color='black', linewidth=2.5)
plt.plot(mem_increase_ANN_3D[1, 0] - vec, mem_increase_ANN_3D[1, 1], '>', color='black', linewidth=2.5)

plt.yscale("log")
plt.xlabel('Accuracy ($R^{2}_{50}$)', fontsize=20, **font_choice, fontweight='bold')
plt.ylabel('Memory (kB)', fontsize=20, **font_choice, fontweight='bold')
plt.xticks(fontsize=16, **font_choice, fontweight='bold')
plt.yticks(fontsize=16, **font_choice, fontweight='bold')
plt.tick_params(axis="both", length=8, width=2, which="major")
plt.tick_params(axis="both", length=4, width=2, which="minor")
plt.legend(handles=handles, labels=labels, loc="best", framealpha=0.5, prop = {'size':13})
ax.spines.top.set_linewidth(3)
ax.spines.bottom.set_linewidth(3)
ax.spines.left.set_linewidth(3)
ax.spines.right.set_linewidth(3)
ax.set_rasterized(True)
plt.savefig("./Memory_vs_Accuracy_50_Split_SNN_ANN.png", bbox_inches='tight')
plt.close()
