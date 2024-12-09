# Watermarking Graph Self-Supervised-Learning

Code for WIFS 2024 paper **"Transferable Watermarking to Self-supervised Pre-trained Graph Encoders by Trigger Embeddings"** https://arxiv.org/abs/2406.13177

# Overview
Our implementation is based on PyTorch and dgl.

**Requirement**
```
dgl                     1.1.1
networkx                3.1
numpy                   1.22.3
scikit-learn            1.3.0
torch                   1.13.1
torch-cluster           1.6.1
torch-geometric         2.3.1
torch-scatter           2.1.1
torch-sparse            0.6.17
torch-spline-conv       1.2.2
torchvision             0.14.1
```
**Please run the following command to play the demo of watermarking of node classification:**
```
python3 wm_train_ready.py 
```
# Acknowledgements
Our code is based on [Graph-Group-Discrimination](https://github.com/zyzisastudyreallyhardguy/Graph-Group-Discrimination), 
thanks for the awesome work.

# Reference

```
@article{zhao2024transferable,
  title={Transferable Watermarking to Self-supervised Pre-trained Graph Encoders by Trigger Embeddings},
  author={Zhao, Xiangyu and Wu, Hanzhou and Zhang, Xinpeng},
  journal={arXiv preprint arXiv:2406.13177},
  year={2024}
}
```