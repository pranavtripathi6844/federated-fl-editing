# Federated Learning with Model Editing on Vision Transformers

This project extends the DINO self-supervised vision transformer framework to support federated learning and model editing. Our implementation enables:

- **Federated Learning (FL):** Training vision transformers (ViT) in a federated setting, supporting both IID and non-IID data distributions across clients.
- **Model Editing:** Integration of gradient masking and sparse updates to study the effects of model editing in both centralized and federated scenarios.
- **Centralized Training:** Standard supervised and self-supervised training of ViT models for comparison with federated approaches.
- **Comprehensive Evaluation:** Scripts for linear probing, k-NN, copy detection, image retrieval, and video segmentation, allowing thorough assessment of learned representations.

The codebase is modular, supporting easy experimentation with different data splits, model architectures, and editing strategies. See the source files for details on running specific experiments.
