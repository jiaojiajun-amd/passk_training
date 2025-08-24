# FROM rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
# FROM vllm-rocm
# FROM rocm/vllm:rocm6.3.1_instinct_vllm0.8.3_20250415
from rocm/vllm:rocm6.4.1_vllm_0.10.0_20250812


# Set working directory
WORKDIR $PWD/app

# Set environment variables
ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942"

# RUN pip install setuptools-scm >=8 
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# ENV ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES


# # Install vllm
# RUN pip uninstall -y vllm && \
#     rm -rf vllm && \
#     git clone -b v0.6.3 https://github.com/vllm-project/vllm.git && \
#     cd vllm && \
#     MAX_JOBS=$(nproc) python3 setup.py install && \
#     cd .. && \
#     rm -rf vllm

# Copy the entire project directory
COPY . .

# Install dependencies
RUN pip install tensordict --no-deps && \
    pip install accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    liger-kernel \
    numpy \
    pandas \
    peft \
    "pyarrow>=15.0.0" \
    pylatexenc \
    "ray[data,train,tune,serve]" \
    torchdata \
    transformers \
    wandb \
    orjson \
    pybind11 && \
    pip install -e . --no-deps

RUN pip install pyvers --no-deps