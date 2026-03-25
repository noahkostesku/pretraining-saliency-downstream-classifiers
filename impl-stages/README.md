The current concrete scaffolding focuses on the core pipeline only:

1. encoder preparation
2. STL-10 split setup and shared downstream wrapper
3. frozen linear-probe training
4. explainability generation
5. explanation faithfulness evaluation

Stages 8 and 9 remain optional and are not part of the main scaffold below until the basics are done.

## Files in this directory

`01-encoder-prep.md`: Covers Stage 1. It defines how to select, load, normalize, freeze, and validate the supervised, MoCo, and SwaV `ResNet-50` checkpoints. It also records checkpoint provenance, shared feature dimensionality, and Grad-CAM layer compatibility.

`02-03-downstream-trainings-and-split.md`: Covers the shared data and model setup before the main experiments. It locks the fixed STL-10 train / validation / test protocol, excludes the unlabeled split from the main study, defines the common preprocessing rules, and specifies the shared downstream wrapper used by both probing and fine-tuning.

`04-trainprobes.md`: Covers the main experiment. It defines frozen linear-probe training for each encoder condition, seed handling, checkpoint selection, reporting requirements, and the minimal sanity checks needed before comparing accuracies.

`05-06-explainability.md`: Covers explanation generation and explanation artifact organization. It sets the target score definition, fixes `encoder.layer4[-1]` as the Grad-CAM target layer, defines the fixed evaluation subset, and describes how Grad-CAM and Occlusion outputs should be saved consistently across conditions.

`07-eval-explain.md`: Covers explanation faithfulness evaluation. It defines the shared perturbation protocol for insertion and deletion AUC, keeps interpretation limited to behavioral faithfulness, and specifies uncertainty reporting with bootstrap confidence intervals if feasible.

`08-opt-fine-tuning.md`: Covers the limited fine-tuning ablation for after the basic training stages are done.

`09-gradCam++.md`: Covers the Grad-CAM++ extension to expand on.

## Execution Order

For each stage, check the following files for more details:

1. `01-encoder-prep.md`
2. `02-03-downstream-trainings-and-split.md`
3. `04-trainprobes.md`
4. `05-06-explainability.md`
5. `07-eval-explain.md`

## How this maps to `plan.md`

- The transfer-learning comparison remains centered on fixed pretrained `ResNet-50` encoders and frozen linear probes.
- The unlabeled STL-10 split is excluded from the main pipeline.
- Explainability is evaluated on a fixed test subset with predicted-class targets.
- Grad-CAM uses `encoder.layer4[-1]`.
- Insertion/deletion AUC uses one shared perturbation protocol across all encoder conditions.
