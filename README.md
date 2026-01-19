# HealthMamba

**HealthMamba: Uncertainty Quantification of Mamba for Multivariate Spatiotemporal Prediction**  
Health facility visit prediction is essential for understanding population health behaviors, optimizing healthcare resource allocation, and guiding public health policy. Despite advanced machine learning models being employed to improve prediction performance, existing approaches rarely consider fine-grained types of health facilities and often suffer performance degradation during public emergencies such as pandemics.

To address these limitations, we propose **HealthMamba**, a spatiotemporal framework for accurate and reliable health facility visit prediction. HealthMamba comprises three key components:
1.  **Unified Spatiotemporal Context Encoder**: Fuses heterogeneous static and dynamic information.
2.  **GraphMamba**: A novel Graph State Space Model for hierarchical spatiotemporal modeling.
3.  **Uncertainty Quantification Module**: Integrates three uncertainty quantification mechanisms for reliable prediction.

We evaluate HealthMamba on four large-scale real-world datasets from California, New York, Texas, and Florida. Results show that HealthMamba achieves approximately **6.0% improvement in prediction accuracy** and **3.5% improvement in uncertainty quantification** over state-of-the-art baselines.

## 🔧 Implementation Details
We conduct experiments on an Quad-Core 2.40GHz – Intel® Xeon X3220, 64 GB RAM linux computing server, equipped with an NVIDIA RTX A100 GPU with 80 GB memory. We adopt PyTorch 2.3.0 and CUDA 11.8 as the default deep learning library.

## 📁 Project Structure

```
├── base/               # Fundamental model and engine
├── src/flow/           # HealthMamba models
├── utils/              # Configuration and dataloader
└── README.md           # Project documentation
```

## 📊 Baselines  

The baselines are inplemented based on [STGCN](https://github.com/hazdzz/STGCN),
[DCRNN](https://github.com/chnsh/DCRNN_PyTorch), 
[DGCRN](https://github.com/wengwenchao123/DDGCRN), 
[StemGNN](https://github.com/microsoft/StemGNN), 
[DSTAGNN](https://github.com/SYLan2019/DSTAGNN), 
[UQGNN](https://github.com/UFOdestiny/UQGNN), 
[ASTGCN](https://github.com/guoshnBJTU/ASTGCN-2019-pytorch), 
[GluonTS](https://github.com/awslabs/gluonts), 
[PatchTST](https://github.com/yuqinie98/PatchTST), 
[ST-LLM](https://github.com/ChenxiLiu-HNU/ST-LLM), 
[UrbanGPT](https://github.com/HKUDS/UrbanGPT), 
[Mamba](https://github.com/state-spaces/mamba), 
[U-Mamba](https://github.com/bowang-lab/U-Mamba), and [AGCRN](https://github.com/LeiBAI/AGCRN).
