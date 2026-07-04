# Animal-10 Image Classification with PyTorch

This project implements a custom GoogLeNet (Inception) inspired Convolutional Neural Network using PyTorch.

## Dataset

Animal-10 Dataset

Classes:

- Butterfly
- Cat
- Chicken
- Cow
- Dog
- Elephant
- Horse
- Sheep
- Spider
- Squirrel

## Model

- Custom Inception Module
- Adam Optimizer
- CrossEntropyLoss
- Image size: 224×224
- Epochs: 20

## Results

Final Test Accuracy: **75.87%**

Train-Test Loss and Accuracy Graph:

![Train-Test Loss and Accuracy Graph](results/Loss-Accuracy.png)

## Prediction Examples

```
butterfly.png -> butterfly
chicken.png -> chicken
cow.png -> dog
```

## Requirements

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```
