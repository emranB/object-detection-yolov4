FROM nvcr.io/nvidia/l4t-tensorrt:r8.6.2-runtime

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    nvidia-cuda-toolkit \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 cache purge
RUN pip3 install opencv-python==4.10.0.84 onnx onnxruntime pycuda
    
# WORKDIR /app/plugings 
# RUN make
# RUN wget https://developer.nvidia.com/embedded/dla/compiler/libnvdla_compiler.so -P /usr/lib/aarch64-linux-gnu/

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/tegra/

COPY ./tensorrt-yolov4 /app

WORKDIR /app/yolo 
RUN python3 yolo_to_onnx.py -m yolov4-416
RUN python3 onnx_to_tensorrt.py -m yolov4-416

WORKDIR /app
CMD python3 rtspyolo.py --rtsp-url rtsp://192.168.1.139:8554/mu -m yolov4-416