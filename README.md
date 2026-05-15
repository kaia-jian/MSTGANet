# MSTGANet
Multi-Scale Spatio-Temporal Graph Attention Network for Wind Vector Forecasting
Accurate and reliable wind vector forecasting is essential for the efficient operation and safe dispatch of wind power systems. MSTGANet is an end-to-end deep learning framework that integrates:

- **Multi-scale 3D convolutional encoder** — extracts hierarchical spatio-temporal features from micro to macro scales
- **Dual-branch adaptive graph learning** — dynamically constructs local and global adjacency matrices to model non-Euclidean wind field structures
- **Spatio-temporal attention mechanism** — enhances focus on critical regions and time steps
- **Sequential decoder** — performs direct multi-step prediction to mitigate error accumulation
- **AGDO hyperparameter optimization** — efficiently searches for optimal model configurations
