# Research Model Improvements

This project keeps the existing EfficientNetV2B0 classifier and improves the training,
evaluation, and deployment pipeline for broader textile-defect generalization.

## Highest-Impact Changes

1. Expand minority defect classes
   - Current training data is imbalanced: `clean` and `fuzzyball` dominate, while
     `contamination`, `knots`, and `weft_crack` have very few samples.
   - Add data from AITEX, TILDA, MVTec AD textures, and Kaggle textile datasets.
   - Map dataset-specific labels into standardized names with `CLASS_ALIASES`.
   - Run `python dataset_tools.py --audit` after every merge to detect duplicates,
     split leakage, and label conflicts.

2. Improve minority-class learning
   - Use clipped square-root class weights instead of extreme inverse-frequency
     weighting.
   - Oversample only rare classes during future experiments if class weights are
     insufficient.
   - Target the weakest classes from `outputs/reports/confusion_pairs.csv`.

3. Strengthen textile-safe augmentation
   - Added brightness, contrast, rotation, translation, zoom, Gaussian noise, and
     light blur.
   - Kept transformations mild and reflection-filled so defects are not warped into
     unrealistic shapes.
   - Avoid aggressive elastic distortion, heavy blur, large rotations, and vertical
     flips unless the fabric orientation is physically irrelevant.

4. Improve fine-tuning
   - Fine-tune the last 80 EfficientNetV2B0 layers instead of only 40.
   - Use AdamW with weight decay.
   - Use lower fine-tuning learning rate: `1e-5`.
   - Use label smoothing to reduce overconfident predictions.

5. Handle unseen defects
   - Added unknown/manual-review logic using confidence and top-class margin.
   - If confidence is low or the top two classes are too close, the app reports
     `Unknown Defect / Manual Review`.
   - For stronger research results, combine this classifier with an anomaly detector
     trained on clean fabric, such as autoencoder reconstruction error, PaDiM,
     PatchCore, or one-class SVM over EfficientNet embeddings.

6. Improve explainability
   - Grad-CAM now averages multiple noisy passes to reduce unstable heatmaps.
   - Heatmaps are smoothed before overlay.
   - In the paper, validate Grad-CAM by checking whether highlighted regions overlap
     visible defects or annotation masks when available.

7. Improve evaluation
   - Evaluation now writes:
     - per-class precision, recall, F1-score
     - binary clean/defect report
     - confusion matrix
     - clean-vs-defect confusion matrix
     - one-vs-rest ROC-AUC where class support permits it
     - most frequent confused class pairs

## Dataset Expansion Workflow

Recommended folder format for new datasets:

```text
external_dataset/
|-- clean/
|-- broken_yarn/
|-- stain/
|-- hole/
`-- weaving_defect/
```

Merge into this project:

```powershell
C:\tdxai_env\Scripts\python.exe dataset_tools.py --merge-root path\to\external_dataset --split train
C:\tdxai_env\Scripts\python.exe dataset_tools.py --audit
```

For AITEX, TILDA, MVTec AD, and Kaggle datasets, first convert each dataset into
folder-per-class format. Then map class names through `CLASS_ALIASES` in
`config.py`.

## Retraining Commands

```powershell
C:\tdxai_env\Scripts\python.exe train_model.py
C:\tdxai_env\Scripts\python.exe evaluate_model.py
```

## Research Paper Framing

Use this as an Explainable AI pipeline:

1. Dataset preparation and leakage audit
2. EfficientNetV2B0 transfer learning
3. Textile-safe augmentation
4. Class-imbalance handling
5. Confidence and unknown-defect decision logic
6. Grad-CAM explainability
7. Quality assessment dashboard

The strongest research contribution is not just classification accuracy; it is the
complete explainable quality-assessment framework with uncertainty-aware decisions.
