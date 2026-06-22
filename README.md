# Combining-SNNs-with-filtering

[![Paper](https://img.shields.io/badge/Paper-IOPscience-blue)](https://iopscience.iop.org/article/10.1088/2634-4386/adba82/meta)
[![Language](https://img.shields.io/badge/Language-Python-3776AB.svg)](https://www.python.org/)

This repository contains the official code implementation for the paper: 
**"Combining SNNs with filtering for efficient neural decoding in implantable brain-machine interfaces"** published by the **BRAINSys Lab** at the **City University of Hong Kong (CityU)**.

---

## 📂 Repository Structure

```text
├── neurobench/         # NeuroBench benchmark implementation (datasets, metrics, and harnesses)
├── main.py             # Main script for offline training and baseline SNN decoding evaluation
├── main_streaming.py   # Script for simulating streaming neural decoding
├── paretoplot_50.py    # Script to plot Pareto frontier (Accuracy vs. Compute/Memory) at 50% density/sparsity
├── paretoplot_80.py    # Script to plot Pareto frontier (Accuracy vs. Compute/Memory) at 80% density/sparsity
```
## 🔧 Installation & Prerequisites

1. Clone the repository:
``` bash
git clone [https://github.com/CityU-BRAINSys-Lab/Combining-SNNs-with-filtering.git](https://github.com/CityU-BRAINSys-Lab/Combining-SNNs-with-filtering.git)
cd Combining-SNNs-with-filtering
```
   
2. Install core dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage:
1. Standard Offline Decoding Evaluation
To run the standard SNN training or evaluation workflow using the offline dataset pipelines:
```bash
python main.py
python main_streaming.py
```

## ✍️ Citation
If you use this code, the benchmarks, or the filtering-SNN methodologies in your academic research, please cite our paper:
  ```text
  @article{biyan2025combining,
    title={Combining SNNs with filtering for efficient neural decoding in implantable brain-machine interfaces},
    author={Biyan, Zhou and Sun, Pao-Sheng Vincent and Basu, Arindam},
    journal={Neuromorphic Computing and Engineering},
    volume={5},
    number={1},
    pages={014013},
    year={2025},
    publisher={IOP Publishing}
  }
  ```


