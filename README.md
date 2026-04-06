# EchoShield — Detecting AI-Generated Voices in the Modern Soundscape

EchoShield is a dual-branch deep learning framework for **AI-generated / deepfake speech detection**, combining **frozen, large pre-trained encoders** with **trainable forensic modules** to improve robustness and generalization across spoofing attacks.

This repository contains:
- The project report (PDF): **EchoShield Detecting AI Generated Voices in the Modern Soundscape.pdf**
- **Kaggle_Local_Training.ipynb**: The fully implemented and improved DDM architecture (≈402M params) featuring 5 major innovations (AACA, DLSN, BCBI, MVCL, Attack-Type Discriminative Branch). 
- **SOTA_DeepFake.ipynb**: A variant tailored for large-scale training on different devices with local dataset environments.
- **Validate_DeepFake.ipynb**: Specifically configured for validation and testing on the ASVspoof 2021 DF evaluation dataset.
- **clap-and-vit-2.ipynb**: A Kaggle-ready training notebook layout.
- Initial dataset and testing scripts (**code.ipynb**, **dataset.ipynb**).

## Highlights

- **Dual-branch architecture (≈402M params)**
  - **Structure Analysis Branch**: frozen **CLAP**, **ViT**, and **Wav2Vec2** encoders for semantic + spectro-temporal + waveform representations
  - **Artifact Detection Branch**: a custom **Artifact Detection Module (ADM)** + temporal modeling (**BiLSTM**) for low-level forensic traces
- **Five core innovations (from the report)**
  1. **ADM**: six parallel forensic extractors (e.g., BayarConv, SRM-style filters) + confidence estimation
  2. **DLSN (Dynamic Layer Selection Network)**: attack-aware weighting/fusion of frozen encoder features
  3. **AACA (Artifact-Aware Cross-Attention)**: uses artifact confidence to modulate attention in the structure branch
  4. **BCBI (Bidirectional Cross-Branch Interaction)**: two-way guidance between structure and artifact branches (redundancy reduction via mutual-information-inspired interaction)
  5. **MVCL (Multi-View Contrastive Loss)**: multi-objective training (focal + contrastive + triplet + consistency terms)

## Reported Results (ASVspoof 2019 LA)

From the report’s evaluation results after **20 epochs** on the **ASVspoof 2019 LA evaluation set**:

- **EER**: **6.8%**
- **Accuracy**: **92.5%**
- **AUC-ROC**: **0.975**
- **Macro F1**: **0.924**

Baselines reported (same evaluation):
- ResNet50 baseline: **11.8% EER**
- Fine-tuned Wav2Vec 2.0: **7.8% EER**

## Dataset

The primary benchmark is **ASVspoof 2019 Logical Access (LA)**.

The report also discusses improving generalization using additional fake sources (e.g., **WaveFake/LJSpeech** and **Release-In-The-Wild**) and recommends cross-dataset validation (e.g., ASVspoof 2021).

## Quickstart (Kaggle & Local)

The primary implementation notebook is **Kaggle_Local_Training.ipynb**, bringing together all the advanced innovations discussed in the report.

### Running on Kaggle
1. Create a Kaggle Notebook and enable GPU.
2. Add the ASVspoof 2019 dataset to the notebook (the notebook references this Kaggle dataset):
   - https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset
3. Open and run **clap-and-vit-2.ipynb** top-to-bottom for the standard Kaggle layout.

### Running Locally
To run the novel DDM implementation locally (**Kaggle_Local_Training.ipynb**), use the following Python packages:

```bash
pip install torch torchvision torchaudio transformers tqdm numpy librosa soundfile scikit-learn matplotlib seaborn
```

If testing ASVspoof 2021 DF specifically, see **Validate_DeepFake.ipynb**. For multi-GPU/larger cluster configurations, refer to **SOTA_DeepFake.ipynb**.

## Figures (Methodology & Performance)

The following diagrams and charts illustrate the EchoShield methodology and the model's training and evaluation performance exactly as presented in the report.

### 1. EchoShield Architecture
![Methodology](Methodology.png)

### 2. Dynamic Layer Selection Network Weight Distribution
![HeatMap](HeatMap.png)

### 3. Artifact Confidence Distributions and Boxplots
![Confidence](Confidence.png)

### 4. Training and Validation Progress
![Train Dev Progress](Train%20Dev%20Progress.png)

## Key Tables (from the report)

### ASVspoof 2019 LA official splits

| Set | Bonafide (Real) | Spoofed (Fake) | Total Samples |
|---|---:|---:|---:|
| Training | 2,580 | 22,800 | 25,380 |
| Development (Validation) | 2,548 | 22,296 | 24,844 |
| Evaluation (Test) | 7,355 | 63,882 | 71,237 |

### Augmented training composition (used for robustness)

| Source Dataset | Type | Samples for Training |
|---|---|---:|
| ASVspoof 2019 LA | Bonafide | 2,580 |
| LJSpeech | Bonafide | 5,000 |
| Release-In-the-Wild (Real) | Bonafide | ≈1,500 |
| **Total Bonafide** |  | **≈9,080** |
| ASVspoof 2019 LA | Spoofed | 22,800 |
| WaveFake | Spoofed | ≈2,000 |
| Release-In-the-Wild (Fake) | Spoofed | ≈1,000 |
| **Total Spoofed** |  | **≈25,800** |
| **Total training samples** |  | **≈34,880** |

### Training hyperparameters (20 epochs)

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Initial learning rate | 1×10^-4 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Total epochs | 20 |
| LR scheduler | CosineAnnealingLR with warmup |
| Warmup epochs | 3 |
| Gradient clipping norm | 1.0 |
| Mixed precision | FP16 (enabled) |
| Focal loss γ | 2.0 |
| NT-Xent temperature τ | 0.07 |
| Loss weights | λ1=1.0 (Focal), λ2=0.3 (Attack), λ3=0.15 (NT-Xent), λ4=0.1 (Triplet), λ5=0.05 (Consistency), λ6=0.1 (MI penalty) |

### Training/validation snapshots

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 | Val AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6234 | 65.2% | 0.5892 | 68.4% | 0.671 | 0.742 |
| 5 | 0.3897 | 83.7% | 0.3845 | 84.8% | 0.846 | 0.909 |
| 10 | 0.2654 | 89.2% | 0.2789 | 88.6% | 0.884 | 0.951 |
| 15 | 0.2012 | 92.1% | 0.2234 | 91.3% | 0.912 | 0.968 |
| 20 | 0.1623 | 93.8% | 0.1987 | 92.7% | 0.926 | 0.977 |

### Evaluation set performance (epoch-20 checkpoint)

| Metric | Value |
|---|---:|
| Equal Error Rate (EER) | 6.8% |
| Overall Accuracy | 92.5% |
| AUC-ROC | 0.975 |
| Macro F1-Score | 0.924 |
| Weighted F1-Score | 0.925 |
| Precision (Bonafide) | 93.1% |
| Recall (Bonafide) | 92.2% |
| Precision (Spoofed) | 92.4% |
| Recall (Spoofed) | 93.3% |

Note: The report abstract also mentions **95.5% accuracy**; the detailed evaluation table above reports **92.5%** after 20 epochs.

### Confusion matrix (evaluation set, 20 epochs)

|  | Predicted Bonafide | Predicted Spoofed | Total |
|---|---:|---:|---:|
| Actual Bonafide | 6,781 (TN) | 574 (FP) | 7,355 |
| Actual Spoofed | 4,280 (FN) | 59,602 (TP) | 63,882 |
| Total | 11,061 | 60,176 | 71,237 |

### Baseline comparison (20 epochs)

| Model | Accuracy | F1-Score | AUC-ROC | EER |
|---|---:|---:|---:|---:|
| ResNet50 (baseline) | 87.3% | 0.871 | 0.934 | 11.8% |
| Wav2Vec 2.0 (fine-tuned) | 91.2% | 0.911 | 0.965 | 7.8% |
| EchoShield (20 epochs) | 92.5% | 0.924 | 0.975 | 6.8% |

## How to Cite

If you reference this work, cite the project report:

> Shahriyar Hasib, Md Mehedi Alam Nahi, Faisal Ahmed, Sahabuddin Shakil. **EchoShield: Detecting AI-Generated Voices in the Modern Soundscape**. B.Sc. thesis/project report, Department of Computer Science and Engineering, United International University, Nov 16, 2025.

## Notes / Limitations

- The report emphasizes generalization; reproducing cross-dataset claims depends on having the corresponding datasets available.
- Training at scale requires GPU resources (the report uses Kaggle GPUs for final runs).
