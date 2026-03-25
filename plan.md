## Research Question

This paper aims to answer:

**Do self-supervised ResNet-50 encoders trained with MoCo and SwaV learn representations that are useful for downstream STL-10 classification, and are the resulting downstream predictions grounded in meaningful image regions when compared against a supervised ResNet-50 baseline encoder?**

A more practical version of the same question is:

**If we freeze each encoder and train the same simple classifier on top, which encoder produces the best representations, and what visual evidence does the final model rely on to make its predictions?**

---

## Implementation Overview

Our implementation answers this question by separating the problem into two parts:

1. **Representation quality**
   We test whether each encoder learned useful features by attaching the same downstream classifier to each encoder and evaluating top-1 accuracy.

2. **Decision evidence / explainability**
   We test whether the final model’s predictions are supported by meaningful image regions by generating explanation maps and evaluating them with deletion and insertion metrics.

The implementation is designed to answer the question **defensibly** by controlling the parts that could otherwise confound the result:

* all encoder conditions use the **same downstream dataset**
* all encoder conditions use the **same downstream classifier architecture**
* the main downstream setting is **linear probing**, so performance mostly reflects encoder quality rather than classifier complexity
* all explanation methods are run on the **same evaluation subset**
* all explanation maps are evaluated under the **same deletion/insertion protocol**
* a **supervised ResNet-50 encoder baseline** is included for comparison

This means that the main difference between conditions is the **encoder pretraining method**, which is exactly what the paper wants to study.

---

# Methods and Processes

## Pipeline Overview

### Stage 1 — Train or prepare the encoders

Train the MoCo encoder, train the SwaV encoder, and prepare a supervised ResNet-50 baseline encoder so that all later downstream comparisons start from three different representation learners.

**Why this works:** each encoder sees images through a different learning objective, so comparing them later isolates the effect of the representation learning method.

### Stage 2 — Train the downstream classifier with linear probing

Freeze each encoder and train only a small linear classification head on STL-10.

**Why this works:** if a simple linear layer performs well, then the encoder representation already contains useful class information.

### Stage 3 — Optional partial fine-tuning

Unfreeze only `layer4` and `fc` for an ablation to test whether slight adaptation changes the downstream result.

**Why this works:** it shows whether the frozen representation is already strong on its own or whether it needs extra task-specific adjustment.

### Stage 4 — Generate explanation maps

Run Grad-CAM and Occlusion on the trained downstream models, with Grad-CAM++ as an optional extra method.

**Why this works:** these methods show which input regions most influenced the model’s class score.

### Stage 5 — Quantify explanation quality

Evaluate the explanation maps using insertion AUC and deletion AUC, and optionally Spearman stability.

**Why this works:** these metrics test whether the regions marked as important actually control the model’s output when inserted or removed.

### Stage 6 — Compare results across encoders

Compare downstream accuracy and explanation metrics for MoCo, SwaV, and the supervised baseline.

**Why this works:** if the downstream head and evaluation protocol are held fixed, differences in performance and saliency behavior can be attributed mainly to the encoder representations.

---

## Experiment Diagram 1 — Encoder Training / Preparation

```text
                 ┌──────────────────────┐
                 │   Raw training data  │
                 └──────────┬───────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        │                   │                   │
        ▼                   ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐
│   MoCo setup   │  │   SwaV setup   │  │ Supervised ResNet-50   │
│ view1 / view2  │  │ multi-crop SSL │  │ baseline encoder       │
│ contrastive    │  │ clustering     │  │ pretrained supervised  │
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

At the end of this stage, all three conditions produce a **ResNet-50 encoder** that maps an input image into a learned feature representation.
The MoCo and SwaV variants differ in **how they learned**, while the supervised baseline differs in **what objective it was trained for**.

---

## Experiment Diagram 2 — Downstream Adaptation by Linear Probe

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
Linear classifier (2048 → 10)
    │
    ▼
STL-10 class logits
    │
    ▼
Top-1 accuracy
```

### Explanation

For the main experiment, the encoder is frozen and only the final linear layer is trained.

This is the **linear probing** setup:

* `requires_grad=False` for encoder parameters
* `requires_grad=True` only for the final `fc` layer

For STL-10, the trainable parameter count should be:

```
2048 \times 10 + 10 = 20490
```

This makes the downstream classifier intentionally simple, which is important because it means good performance mostly reflects the encoder, not a powerful head.

---

## Experiment Diagram 3 — Optional Partial Fine-Tune Ablation

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

This optional ablation unfreezes only:

* `model.layer4`
* `model.fc`

It tests whether a small amount of task-specific adaptation improves performance beyond the frozen linear probe.

This is useful as an extra analysis, but it should remain secondary because the linear probe is the cleaner representation-quality test.

---

## Experiment Diagram 4 — Explainability Pipeline

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
      (main)          (main)          (optional)             (optional)
          │               │                │                     │
          ▼               ▼                ▼                     ▼
   H×W heatmap      H×W heatmap       H×W heatmap        paired heatmaps
          │               │                │
          └───────────────┴────────────────┴──────────────┐
                                                          ▼
                                              Common saliency map format
                                              resize / normalize / save .npy
```

### Explanation

This stage produces explanation maps for the same trained downstream model.

* **Grad-CAM** is the main gradient-based method
* **Occlusion** is the main perturbation-based method
* **Grad-CAM++** is optional
* **Stability** is optional and only compares the consistency of maps

The key design choice is that all methods are converted into the **same `H x W` map format** so they can be compared under the same evaluation protocol.

---

## Experiment Diagram 5 — Explanation Evaluation Metrics

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

These are not explanation methods themselves.
They are **metrics used to score the explanation maps**.

* **Deletion AUC** asks whether removing the most salient regions quickly reduces the model score
* **Insertion AUC** asks whether adding the most salient regions quickly restores the model score

This gives a quantitative notion of faithfulness.

---

## Recommended Core Implementation Order

### 1. Prepare the three encoders

* MoCo-pretrained ResNet-50 encoder
* SwaV-pretrained ResNet-50 encoder
* supervised ResNet-50 baseline encoder

### 2. Implement a shared downstream model wrapper

Each condition should expose the same interface:

```text
image → encoder → 2048-D feature → classifier
```

### 3. Train linear probes for all three encoders

This is the main downstream experiment.

### 4. Report top-1 STL-10 accuracy

This answers whether the learned representations are useful.

### 5. Generate Grad-CAM and Occlusion maps

Do this on the same evaluation subset for all conditions.

### 6. Compute deletion AUC and insertion AUC

This answers whether the explanation maps are behaviorally meaningful.

### 7. Optionally run Grad-CAM++ and stability

These are useful extras but should not dominate the core experiment.

---

## What Each Component Contributes to the Paper

### MoCo / SwaV / supervised baseline encoder comparison

Answers: **which pretraining strategy produces the best representation?**

### Linear probing

Answers: **how much class information is already linearly recoverable from the frozen encoder?**

### Grad-CAM / Occlusion

Answers: **what image regions does the final model rely on?**

### Insertion / Deletion AUC

Answers: **are those highlighted regions actually important to model behavior?**

### Optional partial fine-tuning

Answers: **does the encoder need extra adaptation, or is the frozen representation already strong?**

### Optional Grad-CAM++

Answers: **does a more refined CAM variant change the qualitative or quantitative saliency story?**

### Optional stability

Answers: **are the explanation maps consistent under small perturbations?**

---

## Defensible Experimental Framing

The paper should be careful in how it interprets the results.

A strong, defensible claim is:

> “Under a controlled downstream linear-probe setting, differences in classification accuracy and explanation quality across conditions mainly reflect differences in the learned encoder representations.”

A weaker claim, which you should avoid, is:

> “We fully recovered everything each encoder learned internally.”

This setup does **not** completely decode the encoder’s internal knowledge.
What it does do very well is test:

* whether the representation is useful
* whether a simple classifier can read that usefulness out
* whether the resulting decisions are grounded in plausible and behaviorally important image regions

---

