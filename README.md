<p align="center">
  <img src="assets/misc/phycv_logo.png">
</p>

# PhyCV CUDA: Optimized CUDA Code for Running PhyCV Algorithms on Edge Devices

Welcome to PhyCV CUDA! In this repository, you can find the source code needed to build and run the CUDA versions of the PhyCV algorithms on edge devices like NVIDIA's Jetson Nano. 

## Contents

* [Introduction](#introduction)

* [Folder Structure](#folder-structure)

* [Installation](#installation)

* [Algorithms](#algorithms)

* [Sample Results](#sample-results)

## Introduction 

PhyCV CUDA holds the source files needed to run the C++/CUDA versions of the [PhyCV](https://github.com/JalaliLabUCLA/phycv/) algorithms. The PhyCV CUDA source code can be built and run on any machine with C++/CUDA support. Testing and benchmarks are done on NVIDIA's Jetson Nano. 

## Folder Structure

- `assets`: sample input images/videos, sample results, documentations.
- `phycv_cuda`: source code of PhyCV CUDA.

## Installation

These algorithms use `CUDA` to leverage parallel processing on the GPU. Make sure you have a compatible GPU and that you have set up `CUDA` before the installation. 

**From Source**

```bash
git clone https://github.com/TaejusYee2001/PhyCV_CUDA.git
```
## Algorithms

* Vision Enhancement via Virtual diffraction and coherent Detection (VEViD)

  **Build**
  ```bash
  cd phycv_cuda
  make
  ```
  **Run**
  ```
  ./vevid
  ```
