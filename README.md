# UkrTextRec — Ukrainian Handwritten Text Recognition with CRNN

This project implements a full pipeline for recognizing Ukrainian handwritten text using a Convolutional Recurrent Neural Network (CRNN) model in TensorFlow. It includes data preprocessing, augmentation, model training, evaluation, and inference.

Author: Anna Hnatiuk

## Features

- Custom CRNN architecture with CTC loss for sequence recognition
- Character-level encoding using custom Ukrainian alphabet
- Image preprocessing with dynamic resizing and normalization
- Optional data augmentation (noise, blur, contrast, brightness)
- Dataset split into training, validation, and test sets
- Evaluation using CER (Character Error Rate) and WER (Word Error Rate)
- Visualization of training progress
- Functions for single and batch inference

## Dataset Structure

- `lines/` — contains grayscale PNG images of handwritten text lines
- `METAFILE.tsv` — tab-separated file mapping each image to its ground-truth label
- `alphabet.txt` — list of all unique characters used in the dataset, including space

All files should be placed in the `/content/` directory if using Google Colab.

## Data Preparation

Unzip the `lines.zip` archive to `/content/lines/` and verify that the following files are present:

- `/content/lines/`
- `/content/METAFILE.tsv`
- `/content/alphabet.txt`

These files are used to load and preprocess the data before training.

## Preprocessing

Each image is resized to a fixed height (default: 128 pixels) while preserving aspect ratio. Pixel values are normalized to the range [-1, 1]. Augmentation is applied to 20% of training images and includes:

- Additive Gaussian noise
- Random Gaussian blur
- Random brightness and contrast adjustment

## Model Architecture

The CRNN model consists of:

- 5 convolutional layers with ReLU activation and batch normalization
- Max pooling layers to reduce spatial dimensions
- Reshaping to a sequence for time-distributed processing
- 2 bidirectional LSTM layers for sequence modeling
- Final dense layer with softmax activation
- CTC loss for alignment-free sequence training

## Training

The dataset is split as follows:

- 95% training
- 4% validation
- 1% test

Training uses:

- Batch size: 32
- Epochs: 50
- Optimizer: Adam
- Early stopping with patience = 7
- Best model checkpointing based on validation loss

Training metrics include:

- Loss (CTC)
- CER (Character Error Rate)
- WER (Word Error Rate)

## Evaluation

Evaluation metrics are calculated using Levenshtein distance between predicted and ground-truth sequences. Batch-level CER and WER are computed and logged for analysis.

## Inference

The following functions are provided:

- `predict_image(model, image_path)` — returns the predicted text for a single image
- `predict_many_images(model, image_paths, gt_texts, amount=5)` — prints predictions and CER/WER for randomly sampled images

Example output:

Evaluating 5 randomly selected images...

File: a01-001-0023-03.png
GT: ми любимо свободу
Pred: ми любим свободу
CER: 0.0909
WER: 0.3333

Avg CER: 0.0732
Avg WER: 0.2801


## Requirements

- Python 3.7+
- TensorFlow 2.8+
- NumPy
- Pillow
- Matplotlib
- scikit-learn

## Running in Google Colab

1. Upload the following files:
   - `lines.zip`
   - `METAFILE.tsv`
   - `alphabet.txt`
(dataset with ukrainian handwritten text will soon be published for public use)
2. Run the provided notebook or script step-by-step.
3. Start training and evaluate the model.

## Citation

If you use this project in your research or applications, please cite it as:

@misc{ukrtextrec,
author = {Anna Hnatiuk},
title = {UkrTextRec: Ukrainian Handwritten Text Recognition using CRNN},
year = {2025},
url = {https://colab.research.google.com/drive/1NNWWvWyHYMICofXx0a66TEUjKQP9whPL?usp=sharing}
}


## License

This project is released for academic and educational use only. Please contact the author for other licensing or usage purposes.
