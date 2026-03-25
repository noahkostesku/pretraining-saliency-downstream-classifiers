

## Research Question

This paper aims to answer:

**How do ImageNet-pretrained supervised, MoCo, and SwaV ResNet-50 encoders compare as transferable representations for downstream STL-10 classification under a controlled linear-probe setting, and what evidence do the resulting models use under a fixed explanation protocol?**

A more practical version of the same question is:

**If we freeze each pretrained ResNet-50 encoder and train the same simple classifier on top, which encoder produces the best transferable representation on STL-10, and how behaviorally faithful are the resulting explanation maps under the chosen evaluation protocol?**

---

## Encoder Conditions

All representation-learning conditions use the same backbone family so that the main experimental difference is the pretraining objective rather than the architecture.

### Condition A - Supervised baseline

* architecture: `ResNet-50`
* pretraining source: `ImageNet`
* pretraining objective: `supervised classification`
* checkpoint origin: `public pretrained checkpoint`
* downstream adaptation in the main experiment: `frozen linear probe`

### Condition B - MoCo encoder 

* architecture: `ResNet-50`
* pretraining source: `ImageNet`
* pretraining objective: `MoCo self-supervised contrastive learning`
* checkpoint origin: `public pretrained checkpoint`
* downstream adaptation in the main experiment: `frozen linear probe`

### Condition C - SwaV encoder 

* architecture: `ResNet-50`
* pretraining source: `ImageNet`
* pretraining objective: `SwaV self-supervised clustering`
* checkpoint origin: `public pretrained checkpoint`
* downstream adaptation in the main experiment: `frozen linear probe`

These encoder weights are loaded from public pretrained checkpoints rather than self-trained in this project. This design choice keeps the study focused on transferability under controlled downstream conditions and keeps the compute budget aligned with a defensible baseline study.

---

## Implementation Overview

Our implementation answers the question by separating the problem into two parts:

1. **Representation quality under transfer**
   We test whether each pretrained encoder learned transferable features by attaching the same downstream linear classifier to each encoder and evaluating STL-10 top-1 accuracy.

2. **Faithfulness of explanation maps to model behavior**
   We test whether the regions highlighted by the explanation method are behaviorally influential for the model's target score by generating explanation maps and evaluating them with deletion and insertion metrics.

The implementation is designed to answer the question defensibly by controlling the parts that could otherwise confound the result:

* all encoder conditions use the **same backbone family: ResNet-50**
* all encoder conditions use the **same pretraining source domain: ImageNet**
* all encoder conditions use the **same downstream dataset: STL-10**
* all encoder conditions use the **same downstream classifier architecture**
* the main downstream setting is **linear probing**, so performance mostly reflects encoder quality rather than classifier complexity
* all encoder conditions use the **same train / validation / test protocol**
* all encoder conditions use the **same hyperparameter budget and downstream training recipe**
* all explanation methods are run on the **same evaluation subset**
* all explanation maps are evaluated under the **same deletion / insertion protocol**

This means that the main experimental contrast is the pretrained encoder condition, which differs primarily in pretraining objective but may also reflect checkpoint-specific differences in the pretraining recipes for MoCo, SwAV and the ResNet50 baseline. This is due to using public pretrained checkpoints with differences such as:

- pretraining objective
- checkpoint-specific training recipe
- augmentation policy
- optimizer/schedule
- training duration
- exact checkpoint source / implementation details

---

# Methods and Processes

## Dataset and Split Protocol

The downstream task uses STL-10 as a transfer-learning benchmark.

### Main split usage

* downstream train split: the official STL-10 labeled `train` split, after reserving a fixed stratified holdout for validation
* validation split: a fixed stratified subset carved from the official labeled `train` split
* test split: the official STL-10 labeled `test` split
* unlabeled split: **not used** in the main transfer-learning study

### Why this split is used

The main study is framed as transfer learning rather than in-domain self-supervised pretraining. Excluding the STL-10 unlabeled split keeps the comparison focused on how well pretrained ImageNet representations transfer to STL-10 under the same downstream supervision budget.

### Split control rules

* the same fixed train / validation / test partition is used for all encoder conditions
* the same fixed partition is used across seeds unless a later extension explicitly studies split variability
* model selection is performed on the validation split only
* final performance is reported on the official STL-10 test split only

---

## Pipeline Overview

### Stage 1 - Load the pretrained encoders

Load the supervised, MoCo, and SwaV ResNet-50 checkpoints pretrained on ImageNet so that all later comparisons start from three different pretrained representation learners.

**Why this works:** each encoder uses the same architecture but differs in pretraining objective, which makes the transfer comparison cleaner and much more compute-efficient than training all encoders from scratch.

### Stage 2 - Train the downstream classifier with linear probing

Freeze each encoder and train only a small linear classification head on STL-10.

**Why this works:** if a simple linear layer performs well, then the pretrained encoder representation already contains transferable class information.

### Stage 3 - Optional partial fine-tuning ablation

Unfreeze only `layer4` and `fc` for a secondary ablation to test whether slight adaptation changes the downstream result.

**Why this works:** it shows whether the frozen representation is already strong on its own or whether limited task-specific adaptation changes the ranking across pretrained encoders.

### Stage 4 - Generate explanation maps

Run Grad-CAM as the main explanation method on the trained downstream models, with Occlusion as a secondary perturbation-based comparison and Grad-CAM++ as an optional extra method.

**Why this works:** these methods identify input regions that most influence the model's target score under the chosen explanation protocol.

### Stage 5 - Quantify explanation faithfulness

Evaluate the explanation maps using insertion AUC and deletion AUC, and optionally compute bootstrap confidence intervals for these metrics.

**Why this works:** these metrics test whether the highlighted regions are behaviorally influential for the model score when inserted or removed under the perturbation design.

### Stage 6 - Compare results across encoders

Compare downstream accuracy and explanation metrics for MoCo, SwaV, and the supervised baseline using mean +- std over multiple seeds.

**Why this works:** if the downstream head, split protocol, and evaluation procedure are held fixed, differences in performance and saliency behavior can be attributed mainly to differences in pretrained encoder representations.

---

## Downstream Conditions Held Constant Across Encoders

Each encoder condition should use the same downstream recipe so that only the encoder weights differ.

* input preprocessing: identical across all conditions
* encoder backbone: `ResNet-50`
* feature dimension after pooling: `2048`
* main downstream mode: encoder frozen, only the linear classifier trained
* classifier head: single linear layer `2048 -> 10`
* optimizer, learning-rate schedule, batch size, augmentation, epoch budget, and early-stopping rule: identical across all conditions
* validation model selection rule: identical across all conditions
* seed protocol: identical number of seeds and same reporting format across all conditions

This is the core control that makes the comparison defensible.

---

## Experiment Diagram 1 - Encoder Loading / Preparation

```text
                 ┌─────────────────────────┐
                 │   Public checkpoints    │
                 │ ImageNet-pretrained     │
                 └──────────┬──────────────┘
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
        ▼                   ▼                    ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐
│  MoCo ResNet50 │  │  SwaV ResNet50 │  │ Supervised ResNet-50   │
│ ImageNet SSL   │  │ ImageNet SSL   │  │ ImageNet supervised    │
│ public weights │  │ public weights │  │ public weights         │
└───────┬────────┘  └───────┬────────┘  └──────────┬─────────────┘
        │                   │                      │
        ▼                   ▼                      ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐
│ MoCo-pretrained│  │ SwaV-pretrained│  │ Supervised pretrained  │
│ ResNet-50      │  │ ResNet-50      │  │ ResNet-50 encoder      │
│ encoder        │  │ encoder        │  │                        │
└────────────────┘  └────────────────┘  └────────────────────────┘
```

### Explanation

At the end of this stage, all three conditions produce a pretrained **ResNet-50 encoder** that maps an input image into a learned feature representation. The three variants differ in **how they were pretrained**, not in backbone family, downstream head, or downstream dataset.

---

## Experiment Diagram 2 - Downstream Adaptation by Frozen Linear Probe

```text
Input image
    │
    ▼
Frozen encoder (MoCo / SwaV / Supervised baseline)
    │
    ▼
2048-D pooled feature vector
    │
    ▼
Linear classifier (2048 -> 10)
    │
    ▼
STL-10 class logits
    │
    ▼
Top-1 accuracy
```

### Explanation

For the main experiment, the pretrained encoder is frozen and only the final linear layer is trained.

This is the **linear probing** setup:

* `requires_grad=False` for encoder parameters
* `requires_grad=True` only for the final `fc` layer

For STL-10, the trainable parameter count should be:

```text
2048 x 10 + 10 = 20490
```

This makes the downstream classifier intentionally simple, which is important because it means good performance mostly reflects the transferability of the encoder rather than a powerful head.

---

## Experiment Diagram 3 - Optional Partial Fine-Tune Ablation

```text
Input image
    │
    ▼
Encoder
(layer1, layer2, layer3 frozen)
(layer4 trainable)
    │
    ▼
2048-D pooled feature vector
    │
    ▼
fc layer trainable
    │
    ▼
STL-10 prediction
```

### Explanation

This secondary ablation unfreezes only:

* `model.layer4`
* `model.fc`

It tests whether a small amount of task-specific adaptation improves performance beyond the frozen linear probe. It should remain secondary because the linear probe is the cleaner transfer-representation test.

---

## Experiment Diagram 4 - Explainability Pipeline

```text
Trained downstream model
(encoder + classifier)
          │
          ▼
   Evaluation image
          │
          ├───────────────┬────────────────┬─────────────────────┐
          │               │                │                     │
          ▼               ▼                ▼                     ▼
      Grad-CAM        Occlusion       Grad-CAM++          Optional stability
       (main)        (secondary)       (optional)             (optional)
          │               │                │                     │
          ▼               ▼                ▼                     ▼
    HxW heatmap      HxW heatmap       HxW heatmap        paired heatmaps
          │               │                │
          └───────────────┴────────────────┴──────────────┐
                                                          ▼
                                              Common saliency map format
                                              resize / normalize / save .npy
```

### Explanation

This stage produces explanation maps for the final downstream model.

* **Grad-CAM** is the main gradient-based explanation method
* **Occlusion** is the secondary perturbation-based comparison
* **Grad-CAM++** is optional
* **Stability** is optional and only compares consistency of maps

The key design choice is that all methods are converted into the same `H x W` map format so they can be compared under the same evaluation protocol.

---

## Experiment Diagram 5 - Explanation Evaluation Metrics

```text
Saved saliency map
      │
      ▼
Rank pixels / patches by saliency
      │
      ├──────────────────────────────┐
      │                              │
      ▼                              ▼
Deletion evaluation              Insertion evaluation
start from full image            start from masked image
remove top-ranked regions        add top-ranked regions
track target score               track target score
      │                              │
      ▼                              ▼
Deletion curve                   Insertion curve
      │                              │
      ▼                              ▼
Deletion AUC                     Insertion AUC
(lower is better)               (higher is better)
```

### Explanation

These are not explanation methods themselves. They are **metrics used to score explanation maps**.

* **Deletion AUC** asks whether removing highly ranked regions quickly reduces the model's target score
* **Insertion AUC** asks whether adding highly ranked regions quickly restores the model's target score
* both metrics quantify **faithfulness to model behavior under the chosen perturbation procedure**

These metrics do not establish semantic truth and should not be interpreted as direct evidence of what the model "really understands."

---

## Statistical Reporting and Uncertainty

The study should report performance variability rather than single-run outcomes.

* linear-probe training should be run for a minimum of **3 random seeds** per encoder condition
* top-1 accuracy should be reported as **mean +- standard deviation** across seeds
* validation-based checkpoint selection should be applied independently within each seed
* explanation metrics should be summarized over a fixed evaluation subset
* insertion AUC and deletion AUC should include **bootstrap confidence intervals** if computationally feasible

This study is comparative rather than heavily powered for strong inferential statistics, so the main goal of these uncertainty estimates is to show run-to-run stability and practical effect size.

---

## Design Choices Influenced by Budget

Several choices are intentional simplifications made to keep the methods efficient while still defensible.

* use **public pretrained encoders** instead of self-training MoCo and SwaV from scratch
* use **frozen linear probing** as the main downstream protocol
* use **coarse masking or patchwise perturbation** for insertion / deletion AUC if needed for tractable runtime
* keep **fine-tuning limited to a small ablation** rather than expanding the main study
* keep **optional methods** such as Grad-CAM++ and stability secondary to the core pipeline

These choices reduce compute while preserving the main comparative question.

---

## Recommended Core Implementation Order

### 1. Load the three pretrained encoders

* supervised ImageNet-pretrained ResNet-50
* MoCo ImageNet-pretrained ResNet-50
* SwaV ImageNet-pretrained ResNet-50

### 2. Define the fixed STL-10 split protocol

* fixed stratified train / validation split from STL-10 labeled `train`
* official STL-10 labeled `test` as the held-out test set
* no use of the STL-10 unlabeled split in the main study

### 3. Implement a shared downstream model wrapper

Each condition should expose the same interface:

```text
image -> encoder -> 2048-D feature -> classifier
```

### 4. Train frozen linear probes for all three encoders

This is the main downstream experiment.

### 5. Report top-1 STL-10 accuracy as mean +- std

This answers whether the pretrained representations are useful under transfer.

### 6. Generate Grad-CAM maps & Secondary Occlusions 

Do this on the same evaluation subset for all conditions.

### 7. Compute deletion AUC and insertion AUC

Use the same perturbation protocol for all conditions and include bootstrap confidence intervals if feasible.

### 8. Optionally run the partial fine-tuning ablation

This tests whether limited adaptation changes the main story.

### 9. Grad-CAM++ and stability

These are useful extras but should not dominate the core experiment.

---

## What Each Component Contributes to the Paper

### Supervised / MoCo / SwaV encoder comparison

Answers: **which pretraining objective transfers best to STL-10 under fixed downstream conditions?**

### Linear probing

Answers: **how much class information is linearly recoverable from the frozen pretrained encoder?**

### Grad-CAM

Answers: **which regions most influence the downstream model prediction under the chosen explanation method?**

### Insertion / Deletion AUC

Answers: **are the highlighted regions behaviorally influential for the model's target score under the perturbation protocol?**

### Optional partial fine-tuning

Answers: **does limited task adaptation improve results or change the ranking across encoder conditions?**

### Optional Occlusion / Grad-CAM++ / stability

Answers: **do secondary explanation analyses support or complicate the main Grad-CAM-based interpretation?**

---

## Defensible Experimental Framing

The paper should be careful in how it interprets the results.

A strong, defensible claim is:

> "Under a controlled transfer-learning setup with a fixed ResNet-50 backbone, public ImageNet-pretrained checkpoints, identical STL-10 downstream training, and a common explanation protocol, differences across conditions mainly reflect differences in the transferability of the pretrained representations."

A weaker claim, which you should avoid, is:

> "We fully recovered what each encoder understands internally, and the explanation maps prove semantic truth about the model's reasoning."

This setup does **not** completely decode the encoder's internal knowledge.
What it does do well is test:

* whether the representation transfers usefully to STL-10
* whether a simple classifier can read that usefulness out
* whether highlighted regions are behaviorally influential for the model under the chosen explanation protocol

---
