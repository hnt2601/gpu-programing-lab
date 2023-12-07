# gpu-programing-lab

### Installation
#### CUDA
```bash
sudo docker run --gpus all --name cuda-lab -dit nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04 /bin/bash
```
#### OPENCL
```bash
sudo docker build -t nvidia/opencl .
sudo docker run --gpus all --name opencl-lab -dit nvidia/opencl /bin/bash
```

### Running CUDA
```bash
nvcc main.cu
```

### Running OpenCL
```bash
gcc main.c -lOpenCL -D CL_TARGET_OPENCL_VERSION=300
```