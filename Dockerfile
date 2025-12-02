FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
RUN apt update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"


WORKDIR /workspace

RUN uv venv /opt/venv --python 3.10 && \
    . /opt/venv/bin/activate && \
    uv pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    uv pip install compressai einops timm tensorboard pytorch_msssim 

ENV PATH="/opt/venv/bin:$PATH"
