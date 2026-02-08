# HealthMamba

**HealthMamba: An Uncertainty-aware Spatiotemporal Graph State Space Model for Effective and Reliable Healthcare Facility Visit Prediction**

HealthMamba is a spatiotemporal framework for accurate and reliable healthcare facility visit prediction. It comprises three key components:

1. **Unified Spatiotemporal Context Encoder (STCE)**: Fuses heterogeneous visit data, static node attributes, and dynamic external factors through feature embedding, graph convolution for spatial encoding, and depthwise temporal convolution with channel mixing.

2. **GraphMamba Backbone (G-Mamba)**: A UNet-style architecture integrating adaptive graph learning into State Space Models. Each G-Mamba block performs adaptive graph learning, graph-enhanced spatial mixing, SSM-based temporal mixing, and channel mixing with residual connections. Multi-scale encoding/decoding with skip connections enables hierarchical spatiotemporal modeling.

3. **Uncertainty-aware Prediction**: Combines three complementary uncertainty quantification mechanisms:
   - **Node-based**: Quantile regression heads for prediction intervals (pinball loss)
   - **Distribution-based**: Gaussian heteroscedastic heads for mean and variance (NLL loss)
   - **Parameter-based**: MC Dropout for epistemic uncertainty estimation
   - **Post-hoc calibration**: Conformal quantile calibration on a held-out set

## Project Structure

```
HealthMamba_Code/
├── model/
│   ├── stce.py            # Unified Spatiotemporal Context Encoder
│   ├── gmamba.py           # G-Mamba Block and GraphMamba UNet Backbone
│   └── healthmamba.py      # Full HealthMamba model with uncertainty heads
├── engine/
│   ├── engine.py           # Training engine with unified loss
│   └── metrics.py          # Evaluation metrics
├── utils/
│   ├── args.py             # Argument configuration
│   ├── dataloader.py       # Data loading utilities
│   ├── scaler.py           # Data normalization scalers
│   ├── graph.py            # Graph normalization algorithms
│   └── log.py              # Logger
├── .gitignore
├── LICENSE
└── README.md
```

## Implementation Details

- PyTorch 2.3.0, CUDA 11.8
- Mamba SSM (`mamba_ssm` package)
- NVIDIA RTX A100 GPU (80 GB)

## Baselines

The baselines are implemented based on
[DCRNN](https://github.com/chnsh/DCRNN_PyTorch),
[STGCN](https://github.com/hazdzz/STGCN),
[AGCRN](https://github.com/LeiBAI/AGCRN),
[DGCRN](https://github.com/wengwenchao123/DDGCRN),
[UQGNN](https://github.com/UFOdestiny/UQGNN),
[DSTAGNN](https://github.com/SYLan2019/DSTAGNN),
[ASTGCN](https://github.com/guoshnBJTU/ASTGCN-2019-pytorch),
[GluonTS](https://github.com/awslabs/gluonts),
[PatchTST](https://github.com/yuqinie98/PatchTST),
[ST-LLM](https://github.com/ChenxiLiu-HNU/ST-LLM),
[UrbanGPT](https://github.com/HKUDS/UrbanGPT),
[Mamba](https://github.com/state-spaces/mamba), and
[U-Mamba](https://github.com/bowang-lab/U-Mamba).
