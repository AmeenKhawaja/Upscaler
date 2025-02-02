# Image Upscaler
View the report to learn about how this was implemented!
Deep Learning for Image Super-Resolution
A fun project I worked on that focuses on implementing and analyzing deep learning models to enhance image resolution using Convolutional Neural Networks (CNNs).

## Overview
The goal is to upscale low-resolution images while preserving details and minimizing noise. Three different CNN architectures were implemented and evaluated using the DIV2K dataset, which contains paired high-quality and low-quality images.

The models were trained and evaluated based on Peak Signal-to-Noise Ratio (PSNR), demonstrating how different CNN depths impact super-resolution performance.

## Features
Three CNN architectures: A basic SRCNN-inspired model, a balanced model, and a deep network inspired by VDSR.
Training on the DIV2K dataset, which contains 800 high-resolution images and their corresponding low-resolution versions.
Evaluation using PSNR, a metric for comparing reconstructed image quality.
Image preprocessing with Lanczos resampling to improve input image quality.
Implemented using TensorFlow and Python.

## Dataset
DIV2K dataset:
High-resolution images: 2040x1404 and other resolutions.
Low-resolution images: Downscaled by a factor of 2 (e.g., 1020x702).
Images were resized to 128x128 (high-res) and 64x64 (low-res) for training due to compute limitations.
Model Architectures
Three different CNN architectures were implemented:

## Models
Model 1 (Baseline - SRCNN Inspired)
15-layer CNN with Conv2D, BatchNormalization, PReLU activation, and UpSampling2D.
Moderate feature extraction, balanced between speed and performance.
PSNR Improvement: ~0.6 dB.

Model 2 (Balanced - Deeper Network)
11-layer CNN with higher filter sizes (64, 128) and BatchNormalization layers.
Achieved the best balance between depth and training performance.
PSNR Improvement: ~1.4 dB (Best among the three models).

Model 3 (Deep - VDSR Inspired)
26-layer CNN with increasing/decreasing filter sizes (64 → 256 → 64).
Issues: Overfitting and instability (artifacts in output).
PSNR Improvement: ~0.5 dB (Worse than Model 2).
