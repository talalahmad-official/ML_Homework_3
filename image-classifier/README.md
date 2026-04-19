# 🖼️ Image Classifier (CIFAR-10 CNN)

This project implements a **Convolutional Neural Network (CNN)** for image classification using PyTorch. It includes complete pipelines for **data preprocessing, training, evaluation, and experimentation** on a CIFAR-10–style dataset.

---

## 📁 Project Structure

```
image-classifier/
│
├── main.py                  # Entry point for running experiments
├── requirements.txt        # Project dependencies
│
├── dataset/
│   ├── raw_dataset.py      # Converts CSV + images into class folders
│   ├── custom_dataset.py   # Dataset class + train/test split logic
│
├── models/
│   └── model.py            # CNN architecture
│
├── train/
│   └── train.py            # Training loop
│
├── utils/
│   └── helper.py           # Evaluation + plotting utilities
│
├── plots/
│   ├── exp_1_results.png
│   ├── exp_2_results.png
│   └── exp_3_results.png
│
└── results/
    └── best_model.pth      # Saved best model
```

---

## 🚀 Features

- Custom CNN architecture for image classification
- Data preprocessing and dataset organization
- Train/test split automation
- Multiple experiment configurations
- Model evaluation with accuracy and loss
- Visualization of training performance
- Saving best-performing model

---

## 🧠 Model Architecture

The CNN consists of:
- 3 Convolutional layers with ReLU activation
- MaxPooling layers for downsampling
- Fully connected layers for classification

Input shape: **(3 × 32 × 32)**  
Output: Class probabilities (CIFAR-10 style)

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

### Step 1: Organize Raw Dataset

```python
create_mapped_folder(csv_path, src_img_dir, dest_root_dir)
```

### Step 2: Train/Test Split

```python
split_raw_into_final(src_root, dest_root, train_ratio=0.8)
```

---

## 🏃 Running the Project

```bash
python main.py
```

---

## 🧪 Experiments

| Experiment | Epochs | Optimizer | Learning Rate | Batch Size |
|-----------|--------|-----------|--------------|------------|
| 1         | 5      | SGD       | 0.01         | 32         |
| 2         | 10     | Adam      | 0.001        | 64         |
| 3         | 20     | Adam      | 0.001        | 64         |

---

## 📉 Evaluation

```python
evaluate_model(model, test_loader, criterion, device)
```

Outputs:
- Test Loss
- Test Accuracy

---

## 📊 Visualization

```python
plot_training_history(history)
```

Plots are saved in the `/plots` directory.

---

## 💾 Model Saving

```
results/best_model.pth
```

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- Pandas
- Matplotlib
- Pillow

---

## 📌 Workflow

1. Prepare dataset
2. Split into train/test
3. Train CNN
4. Evaluate
5. Visualize results
6. Save best model

---

## ✍️ Author

**Talal Ahmad (MSCS25015)**
