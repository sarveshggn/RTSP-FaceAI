# Face Detection & Recognition System

## Overview
This repository contains a face detection and recognition system built using **OpenVINO**, **FAISS**, and supporting tools.  
The system performs:

- **Face Detection** using **SCRFD**  
- **Face Recognition** using **AdaFace**  
- **Similarity Search** using **FAISS**  

Optimized inference is achieved with Intel CPUs, GPUs, and NPUs.

### Processing Modes
The system supports multiple processing modes:

- **Image Comparison**: Compare two images for face similarity
- **Pre-recorded Video Processing**: Process video files with face detection and recognition
- **RTSP Stream Processing**: Real-time processing of live RTSP camera streams (**30 FPS**)
  - Multi-threaded architecture for parallel capture and processing
  - Automatic reconnection handling for network interruptions
  - Optimized frame buffering for minimal latency (~33ms)

---

## Technologies Used

### 1. OpenVINO
Used for optimized inference of detection and recognition models. Supports acceleration on **Intel CPUs, GPUs, NPUs**.

#### Models:
- **SCRFD (Face Detection)**  
  - Model: `det_10g.onnx` → converted to OpenVINO IR  
  - Input: `(1, 3, 360, 640)`  
  - Output: bounding boxes, keypoints, confidence  
  - Includes blur detection via Laplacian variance  

- **AdaFace (Face Recognition)**  
  - Model: `adaface_r50.xml` (R50 architecture)  
  - Input: `(1, 3, 112, 112)`  
  - Output: `(1, 512)` (face embedding)  
  - Embeddings used for similarity search  

---

### 2. FAISS
Used for efficient similarity search of embeddings.

- **IndexFlatIP** – small datasets, cosine similarity  
- **IndexIVFFlat** – large datasets, `nlist=1024`, `nprobe=100`  
- **IndexHNSWFlat** – fast graph-based search (`M=32`)  

---

### 3. OpenCV
Handles preprocessing, visualization, and video processing:
- Resizing  
- Normalization  
- Drawing bounding boxes
- RTSP stream capture and management  

---

### 4. Python
Core implementation language.  
Dependencies: `numpy`, `opencv-python`, `faiss`, `openvino`, `argparse`, `json`, `threading`, `queue`, `sqlite3`.

---

## Hardware Details
- **CPU**: Intel(R) Core(TM) Ultra 5 125H (18 cores)  
- **Memory**: 32 GB 
- **GPU**: Intel(R) Graphics (iGPU)  
- **NPU**: Intel(R) AI Boost (Meteor Lake and newer)  
- **OS**: Ubuntu 22.04 (Kernel 6.8.0-40-generic)  

---

## Workflow

### 1. Face Detection (SCRFD)
- Resize image to `(640, 360)`  
- Run detection model  
- Extract bounding boxes and keypoints  
- Apply blur filter (`blur_threshold = 25–50`)  

### 2. Face Recognition (AdaFace)
- Crop face and resize to `(112, 112)`  
- Normalize pixels to `[-1, 1]`  
- Run model to generate `(1, 512)` embedding  

### 3. Similarity Search (FAISS)
- Normalize embeddings with L2  
- Compare using cosine similarity  
- Retrieve top matches  

---

## Performance Metrics

| Component           | Time (avg)        | Notes |
|---------------------|------------------|-------|
| Face Detection      | 10–15 ms/frame   | High accuracy frontal faces |
| Face Recognition    | 5–10 ms/face     | 512-dim embeddings |
| FAISS Search        | 1–10 ms/query    | Depends on index type |
| Frame Total         | 50–60 ms         | ~15–20 FPS (pre-recorded videos) |

### Similarity Thresholds
- `>= 0.45` → Same person  
- `< 0.45` → Different person  

### Backend Performance (Pre-recorded Videos)
- **ONNX Runtime** → ~4–5 FPS  
- **OpenVINO + Flat Index** → ~8–10 FPS  
- **OpenVINO + IVFFlat** → ~15–19 FPS (best trade-off)  
- **OpenVINO + HNSW** → ~17–21 FPS (lower accuracy)

### RTSP Stream Performance
- **Average FPS**: **30 FPS** (87–114% improvement over pre-recorded)
- **Frame Latency**: ~33ms per frame (53% reduction)
- **Architecture**: Multi-threaded (parallel capture and processing)
- **Frame Drop Rate**: 5–10% (acceptable trade-off for real-time performance)
- **CPU Utilization**: 40–50% (efficient resource usage)

**Key Optimizations:**
- Minimal buffer size (`CAP_PROP_BUFFERSIZE = 1`) for low latency
- Intelligent frame skipping (drops old frames when queue is full)
- Parallel processing architecture (capture thread + processing thread)
- Optimized FAISS index (IVF with `nprobe=100`)

See [RTSP_STREAM_DOCUMENTATION.md](RTSP_STREAM_DOCUMENTATION.md) for detailed technical documentation.  

### Tunable Parameters
- `blur_threshold` → Blur filtering  
- `nlist`, `nprobe` → IVF index parameters  
- `M`, `dim` → HNSW index parameters  
- `cosine_threshold` → Face match similarity cutoff  

---

## File Structure
```
/home/sr/ov_fr/
│── buffalo_l/ # Model files (SCRFD, AdaFace in ONNX & OpenVINO IR)
│── database/ # Embeddings, database scripts, original images
│── fd/ # Face Detection (SCRFD + blur detection)
│── fr/ # Face Recognition (AdaFace ONNX & OpenVINO)
│── utils/ # Utilities (embedding export, JSONL update, testing)
│── videos/output/ov/ # Results (annotated videos)
├── No_blur/
├── 25_blur/
└── 50_blur/
```

---

## Usage Examples

### RTSP Stream Processing
```python
from main import compare_rtsp_stream

# Basic usage
compare_rtsp_stream(
    rtsp_url='rtsp://192.168.10.94/live1.sdp',
    threshold=0.49,
    jsonl_path='final_embeddings_adaface_ov_fp16.jsonl',
    load_last_n=100000
)

# With recording enabled
compare_rtsp_stream(
    rtsp_url='rtsp://username:password@192.168.10.94:554/stream',
    threshold=0.49,
    jsonl_path='final_embeddings_adaface_ov_fp16.jsonl',
    load_last_n=100000,
    src_folder='/path/to/reference/images',
    enable_recording=True,
    output_path='rtsp_output.mp4'
)
```

### Pre-recorded Video Processing
```python
from main import compare_video_last_detailed

compare_video_last_detailed(
    video_path='input_video.mp4',
    threshold=0.49,
    output_path='output_video.mp4',
    jsonl_path='final_embeddings_adaface_ov_fp16.jsonl',
    load_last_n=100000
)
```

### Image Comparison
```python
from main import compare_two_images
import argparse

args = argparse.Namespace(img1='image1.jpg', img2='image2.jpg')
similarity, conclusion = compare_two_images(args)
print(f"Similarity: {similarity:.3f}, Conclusion: {conclusion}")
```

---

## Sample Results
Annotated output videos are saved in:
- `/home/sr/ov_fr/videos/output/ov/`  
- Google Drive (organized by blur threshold)  

---

## Conclusion
This system integrates **SCRFD**, **AdaFace**, and **FAISS** with **OpenVINO acceleration** for efficient face recognition.  
It demonstrates real-time performance with scalable embedding search and tunable accuracy-speed trade-offs.

**Key Features:**
- Real-time RTSP stream processing at **30 FPS**
- Multi-threaded architecture for optimal performance
- Automatic reconnection handling for robust operation
- Support for pre-recorded videos and live streams
- Efficient FAISS-based similarity search (scalable to millions of embeddings)

---

## Documentation
- **[RTSP_STREAM_DOCUMENTATION.md](RTSP_STREAM_DOCUMENTATION.md)**: Comprehensive guide to RTSP stream processing, architecture, and performance optimizations
- **[fd_fr_solution_document.md](fd_fr_solution_document.md)**: Technical details about models, workflows, and system architecture

## References
- [OpenVINO](https://docs.openvino.ai/)  
- [SCRFD](https://github.com/deepinsight/insightface)  
- [AdaFace](https://github.com/mk-minchul/AdaFace)  
- [FAISS](https://github.com/facebookresearch/faiss)  

