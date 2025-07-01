#                                          88
#   ,d                                     88
#   88                                     88
# MM88MMM ,adPPYba,  8b,dPPYba,  ,adPPYba, 88,dPPYba,
#   88   a8"     "8a 88P'   "Y8 a8"     "" 88P'    "8a
#   88   8b       d8 88         8b         88       88
#   88,  "8a,   ,a8" 88         "8a,   ,aa 88       88
#   "Y888 `"YbbdP"'  88          `"Ybbd8"' 88       88

FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
ARG REQUIREMENTS_FILE
RUN apt update -y
RUN apt-get update -y
RUN mkdir -p ./tmp

COPY ./requirements.txt ./tmp/requirements.txt
WORKDIR ./tmp

ENV FORCE_CUDA="1" \
    MMCV_WITH_OPS=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX;8.9;9.0" 
RUN pip install -r ./requirements.txt
# overwrite triton install (was the only bugfree one.)
RUN pip install triton==2.1.0
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 wget -y
COPY ./install_prebuild_ssm.sh ./install_prebuild_ssm.sh
RUN chmod +x ./install_prebuild_ssm.sh
RUN ./install_prebuild_ssm.sh
WORKDIR /workspace
