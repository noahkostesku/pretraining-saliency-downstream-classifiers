The current concrete scaffolding focuses on the core pipeline:

1. encoder preparation for pretrained conditions
2. STL-10 split setup and shared downstream wrapper
3. main downstream training (frozen probes + random-init full training)
4. explainability generation
5. explanation faithfulness evaluation

Stage 8 remains optional. Stage 9 is now only for supplemental diagnostics after the core explainability pipeline is complete.

## Files in this directory

`stage01/01-encoder-prep.md`: Covers Stage 1. It defines how to select, load, normalize, freeze, and validate the supervised, MoCo, and SwaV `ResNet-50` checkpoints. It also records checkpoint provenance, shared feature dimensionality, and Grad-CAM layer compatibility.

`02-03-downstream-trainings-and-split.md`: Covers the shared data and model setup before the main experiments. It locks the fixed STL-10 train / validation / test protocol, excludes the unlabeled split from the main study, defines the common preprocessing rules, and specifies the shared downstream wrapper used by both probing and fine-tuning.

`04-trainprobes.md`: Covers the main downstream experiment. It defines frozen linear-probe training for supervised, MoCo, and SwaV plus a fully trained random-init `ResNet-50` baseline, with shared seed handling, checkpoint selection, reporting requirements, and sanity checks.

`05-06-explainability.md`: Covers explanation generation and explanation artifact organization. It sets the target score definition, fixes `encoder.layer4[-1]` as the target layer for Grad-CAM and Grad-CAM++, defines one fixed evaluation subset without correctness filtering, and describes how Grad-CAM, Grad-CAM++, and Occlusion outputs should be saved consistently across conditions.

`07-eval-explain.md`: Covers explanation faithfulness evaluation. It defines the shared perturbation protocol for insertion and deletion AUC, keeps interpretation limited to behavioral faithfulness, and specifies uncertainty reporting with bootstrap confidence intervals if feasible.

`08-opt-fine-tuning.md`: Covers the limited fine-tuning ablation for after the basic training stages are done.

`09-gradCam++.md`: Covers optional supplemental diagnostics that build on top of the already-required Grad-CAM++ pipeline.

## Execution Order

For each stage, check the following files for more details:

1. `stage01/01-encoder-prep.md`
2. `02-03-downstream-trainings-and-split.md`
3. `04-trainprobes.md`
4. `05-06-explainability.md`
5. `07-eval-explain.md`
