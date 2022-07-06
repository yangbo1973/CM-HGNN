# CM-HGNN

## Code

This is the source code for KBS 2022 Paper: Category-aware Multi-relation Heterogeneous Graph Neural Networks for session-based recommendation

## Requirements

- Python 3
- PyTorch >= 1.3.0
- tqdm

## Usage

Data preprocessing:

The code for session data preprocessing can refer to [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN).
In addition, we extracted category labels from the three original datasets and recorded them in category.txt. The source of the original datasets can be found in the paper

Train and evaluate the model:
~~~~
python main.py --dataset diginetica
~~~~

## Citation

~~~~
@article{xu2022category,
  title={Category-aware Multi-relation Heterogeneous Graph Neural Networks for session-based recommendation},
  author={Xu, Hao and Yang, Bo and Liu, Xiangkun and Fan, Wenqi and Li, Qing},
  journal={Knowledge-Based Systems},
  pages={109246},
  year={2022},
  publisher={Elsevier}
}
~~~~
