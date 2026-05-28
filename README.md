# GPU-Accelerated Gaussian Blur for Batch Image Processing

## Project Overview

This project demonstrates GPU-accelerated image processing using CUDA programming in Python. The application applies Gaussian Blur filtering to a large dataset of grayscale images and compares CPU execution time with GPU-based parallel execution.

The project was developed and tested using Google Colab with NVIDIA Tesla T4 GPU hardware.

---

## Objective

The objective of this project is to demonstrate how GPU parallel computing can significantly improve performance for image processing workloads involving large datasets.

The project processes:

- 100 grayscale images
- Batch image filtering
- CPU vs GPU benchmarking
- CUDA-based parallel image processing

---

## Technologies Used

- Python
- CUDA
- Numba CUDA
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- Google Colab

---

# GPU Hardware

The project was executed using NVIDIA Tesla T4 GPU on Google Colab.

Check GPU availability:

```python
!nvidia-smi
```

---

# Project Workflow

1. Load image dataset
2. Convert images to grayscale
3. Generate batch dataset of 100 images
4. Apply Gaussian Blur using CPU
5. Apply Gaussian Blur using CUDA GPU kernel
6. Compare execution times
7. Visualize outputs and performance results

---

# Gaussian Blur

Gaussian Blur is an image smoothing technique used to reduce image noise and detail. It works by applying a weighted convolution filter across neighboring pixels.

The Gaussian kernel used:

```text
1  2  1
2  4  2
1  2  1
```

Normalized by dividing by 16.

---

# Installation

Open Google Colab:

https://colab.research.google.com

Enable GPU:

```text
Runtime → Change runtime type → GPU
```

Install required libraries:

```python
!pip install numba pillow matplotlib requests
```

---

# Full Project Code

## Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import requests
import time
from numba import cuda
```

---

## Create Dataset of 100 Images

```python
url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb"

response = requests.get(url)

base_img = Image.open(BytesIO(response.content)).convert("L")

base_img = base_img.resize((512, 512))

num_images = 100

images = []

for i in range(num_images):

    img_array = np.array(base_img).astype(np.float32)

    img_array = np.roll(img_array, shift=i, axis=1)

    images.append(img_array)

images_np = np.stack(images)

print("Dataset Shape:", images_np.shape)
```

---

## CPU Gaussian Blur

```python
def gaussian_blur_cpu(image):

    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16.0

    height, width = image.shape

    output = np.zeros_like(image)

    for y in range(1, height - 1):

        for x in range(1, width - 1):

            value = 0.0

            for ky in range(3):

                for kx in range(3):

                    pixel = image[y + ky - 1, x + kx - 1]

                    value += pixel * kernel[ky, kx]

            output[y, x] = value

    return output
```

---

## Run CPU Processing

```python
start_cpu = time.time()

cpu_results = []

for i in range(num_images):

    result = gaussian_blur_cpu(images_np[i])

    cpu_results.append(result)

cpu_results = np.stack(cpu_results)

end_cpu = time.time()

cpu_time = end_cpu - start_cpu

print("CPU Time:", cpu_time, "seconds")
```

---

## CUDA Gaussian Blur Kernel

```python
@cuda.jit
def gaussian_blur_gpu(images, output):

    img_id, y, x = cuda.grid(3)

    num_imgs, height, width = images.shape

    if img_id < num_imgs and 1 <= y < height - 1 and 1 <= x < width - 1:

        value = 0.0

        value += images[img_id, y-1, x-1] * 1
        value += images[img_id, y-1, x] * 2
        value += images[img_id, y-1, x+1] * 1

        value += images[img_id, y, x-1] * 2
        value += images[img_id, y, x] * 4
        value += images[img_id, y, x+1] * 2

        value += images[img_id, y+1, x-1] * 1
        value += images[img_id, y+1, x] * 2
        value += images[img_id, y+1, x+1] * 1

        output[img_id, y, x] = value / 16.0
```

---

## Run GPU Processing

```python
d_images = cuda.to_device(images_np)

d_output = cuda.device_array_like(images_np)

threads_per_block = (4, 16, 16)

blocks_per_grid = (
    int(np.ceil(num_images / threads_per_block[0])),
    int(np.ceil(images_np.shape[1] / threads_per_block[1])),
    int(np.ceil(images_np.shape[2] / threads_per_block[2]))
)

gaussian_blur_gpu[
    blocks_per_grid,
    threads_per_block
](d_images, d_output)

cuda.synchronize()

start_gpu = time.time()

gaussian_blur_gpu[
    blocks_per_grid,
    threads_per_block
](d_images, d_output)

cuda.synchronize()

end_gpu = time.time()

gpu_results = d_output.copy_to_host()

gpu_time = end_gpu - start_gpu

print("GPU Time:", gpu_time, "seconds")

print("Speedup:", cpu_time / gpu_time)
```

---

## Display Results

```python
index = 0

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(images_np[index], cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cpu_results[index], cmap="gray")
plt.title("CPU Gaussian Blur")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gpu_results[index], cmap="gray")
plt.title("GPU Gaussian Blur")
plt.axis("off")

plt.show()
```

---

## Performance Chart

```python
labels = ["CPU", "GPU"]

times = [cpu_time, gpu_time]

plt.bar(labels, times)

plt.ylabel("Execution Time (seconds)")

plt.title("CPU vs GPU Gaussian Blur Performance")

plt.show()
```

---

## Final Summary

```python
print("Final Project Summary")
print("---------------------")

print("Number of Images Processed:", num_images)

print("Image Resolution:",
      images_np.shape[1], "x",
      images_np.shape[2])

print("CPU Time:", cpu_time, "seconds")

print("GPU Time:", gpu_time, "seconds")

print("Speedup:", cpu_time / gpu_time)
```

---

# Example Output

```text
CPU Time: 12.42 seconds
GPU Time: 0.31 seconds
Speedup: 40x
```

Performance may vary depending on GPU hardware and image size.

---

# CUDA Concepts Used

- CUDA kernels
- Parallel thread execution
- Thread blocks and grids
- Device memory allocation
- GPU acceleration for image filtering
- Parallel batch image processing

---

# Learning Outcomes

This project demonstrates:

- GPU acceleration using CUDA
- Parallel image processing
- CPU vs GPU benchmarking
- CUDA memory management
- Batch image filtering using GPU computation

---

# Future Improvements

Possible future enhancements:

- Real-time video filtering
- Webcam processing
- Additional filters:
  - Sharpening
  - Edge detection
  - Histogram equalization
- Shared memory optimization
- Multi-GPU processing
- Real-time GPU image processing dashboard

---
#OUTPUT
<img width="1176" height="387" alt="image" src="https://github.com/user-attachments/assets/23f89db3-5b3e-40e0-945a-6549fec4eac7" />

---


# Conclusion

This project demonstrates how CUDA-based GPU programming can dramatically accelerate image processing workloads. By processing 100 images in parallel using GPU kernels, the project showcases the advantages of parallel computing for large-scale image filtering tasks.

---

