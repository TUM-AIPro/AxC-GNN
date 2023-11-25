# AxC-GNN
This repository provides a code for the paper [Approximation- and Quantization-Aware Training for Graph Neural Networks](https://www.researchgate.net/publication/373697376_Approximation-Aware_and_Quantization-Aware_Training_for_Graph_Neural_Networks).

## Dependancies
This code requires [PyTorch](https://pytorch.org/) and [PyG](https://pyg.org/) with CUDA. The code was tested on multiple versions of PyTorch (1.7, 2.0, 2.1) and CUDA (11.3, 11.7, 12.1). Note that different versions of PyG may support only certain PyTorch versions. Additional required packages are `sklearn` and `tqdm`.

## CUDA kernel installation
Navigate to `code/cuda/approxmatmul` and run `install_kernel.sh` to install approximate multiplication kernel for [EvoApproxLib](https://ehw.fit.vutbr.cz/evoapproxlib/?folder=multiplers/8x8_signed). If you are using Conda, remove `--user` flag from the `install_kernel.sh` as per Conda recommendation. You can also use:
```bash
pip install .
```
## Run GNN training and evaluation
The sample commands for GNN training are given in `train_gcn_pubmed.sh`. To explore all training options use:
```bash
python main.py --help
```
Note that saved models will be overwritten in their corresponding save folders in `models/...`. Currently training times for quantized networks are larger than for their FP32 versions, since DegreeQuant quantization is performed in numpy on CPU as in the original DegreeQuant [repository](https://github.com/camlsys/degree-quant). However, approximate multiplication adds no additional time overhead for quantized networks.
The sample commands for GNN evaluation can be found in `eval_*.sh`. Mainly, evaluation mode is initiated with `--evalonly` flag for `main.py`.

## Modifying CUDA kernel for custom approximate operations
To test other approximate operations, CUDA kernel needs to be modified and installed again. It is recommended to copy the `code/cuda/approxmatmul` folder and change an extension name in `setup.py`. The only file, that needs to be modified is `apxop_kernels.cu`, in particular `matmul_cuda_kernel` or/and `batch_matmul_cuda_kernel` fucntions within the file. `matmul_cuda_kernel` in `apxop_kernels.cu` contains comments, describing the process of adding new approximate operations. Behavioral C-code of approximate operations can be employed in CUDA directly as `__device__` functions and called from `__global__ void` kernel functions (see examples and comments in `apxop_kernels.cu`).

## Citing the paper
The following paper needs to be cited: 
Rodion Novkin, Florian Klemme, and Hussam Amrouch "Approximationaware and quantization-aware training for graph neural networks", IEEE Transactions on Computers (TC), 2023.
