## Future Extensions

These are the natural next steps to answer deeper questions such as:

**“What is encoded internally?”**

### 1. Representation probing beyond linear classification

Train probes on intermediate layers, not just the final pooled feature.

This can answer:

* where class information appears in the network
* whether different encoders become semantically useful at different depths

### 2. Concept-level probing

Test whether specific visual concepts are encoded, such as:

* texture
* shape
* color
* background
* object parts

This can answer:

* what kinds of concepts the encoder has organized internally

### 3. Nearest-neighbor retrieval in feature space

For a given image embedding, retrieve the closest training examples in encoder space.

This can answer:

* whether the encoder clusters by semantic object category
* whether it relies on background shortcuts
* whether SSL and supervised encoders organize the space differently

### 4. Activation maximization / feature visualization

Visualize what individual channels or neurons respond to most strongly.

This can answer:

* what visual patterns activate different encoder units

### 5. Dataset activation search

Find the images or patches that maximally activate specific channels.

This can answer:

* whether a channel corresponds to object parts, textures, or spurious background features

### 6. Compare explanation maps across encoder layers

Instead of only explaining the final classifier decision, examine how attention shifts from earlier to later layers.

This can answer:

* how representation focus evolves through the encoder hierarchy

### 7. Robustness and counterfactual tests

Test the model under background changes, crop perturbations, or synthetic edits.

This can answer:

* whether the encoder learned robust semantic features or fragile shortcuts

### 8. Segmentation-overlap style localization metrics

If pixel or box annotations are available, compare explanation maps against object masks.

This can answer:

* whether the highlighted regions align with the actual object rather than irrelevant context

---

## Final Summary

A concise implementation plan for the paper is:

* prepare **three encoders**: MoCo, SwaV, supervised ResNet-50 baseline
* evaluate each encoder with the **same downstream linear probe**
* measure **top-1 accuracy**
* generate **Grad-CAM** and **Occlusion** explanation maps
* score those maps using **deletion AUC** and **insertion AUC**
* keep **Grad-CAM++**, **partial fine-tuning**, and **stability** as optional extras
* interpret differences across conditions as evidence about the usefulness and visual grounding of the encoder representations

If you want, I can turn this into a more formal **paper prose draft** or a **thesis proposal style section** next.
