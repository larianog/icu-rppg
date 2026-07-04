# ICU-rPPG

ICU-rPPG is a pilot study comparing deep learning models for real-time heart rate estimation from a recorded video of a single ICU patient.

For more information about the evaluated models and the Open-rPPG toolbox, visit the [open-rPPG](https://github.com/KegangWangCCNU/open-rppg) repository.

## Installation

This repository uses Python 3.12.13.

The recommended way to install the required dependencies is by creating a Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

Then, activate the environment:

```bash
conda activate icu_rppg
```

For more information about installing Conda, see the [Conda installation documentation](https://www.anaconda.com/docs/getting-started/installation).

## Quick Start

The main script in this repository is `inference_video_segments.py`. Before running it, configure the following variables:

- `VIDEO_NAME`: Path to the input video. The video must be located in the `video/` directory.
- `GT_FILE`: Path to the ground truth file. The file must be located in the `ground_truth/` directory.
- `SEGMENT_DURATION`: Duration (in seconds) of each processing window.
- `RESULTS_DIR`: Directory name where the results will be saved.

### Input Requirements

The ground truth file must satisfy the following requirements:

- It must be a `.txt` file.
- The first line must contain the heart rate measurements, with each value separated by a whitespace.
- The second line must contain the corresponding timestamps, also separated by whitespace.

Example:

```text
101.0 102.3 99.9 80.5
0 1.0 2.0 3.0
```

An example ground truth file used in this study is provided in the `ground_truth/` directory.

### Outputs

The script generates two outputs:

1. A comparison plot of the ground truth and estimated heart rate. Example plots are available in `results/graphics/`.
2. A `.txt` file containing the inference time for each processing window, the error for each window, the mean absolute error (MAE), and the Pearson correlation coefficient. A `.txt` file is generated for each model chosen. Examples are available on `results/`.

## Results

The script `inference_video_segments.py` was executed with the following parameters to evaluate heart rate estimation for a single ICU patient:

- `SEGMENT_DURATION = 15`
- `RESULTS_DIR = results/`

Samples from the video used for inference are shown below.

<table>
  <tr>
    <td align="center">
      <img src="image1.png" width="400"><br>
      <b>Model A</b>
    </td>
    <td align="center">
      <img src="image2.png" width="400"><br>
      <b>Model B</b>
    </td>
  </tr>
</table>

The inference results for the evaluated video are available in the `results/` directory. This folder contains detailed output files for each model, including quantitative performance metrics and the corresponding comparison plots.

## Models
The following models are used in this experiment:

| Model Name | Description | Reference |
| :--- | :--- | :--- |
| **ME-chunk** | Temporal-Spatial State Space Duality | arXiv 2025 |
| **ME-flow** | Temporal-Spatial State Space Duality | arXiv 2025 |
| **RhythmMamba**| Multi-Temporal Constraint Mamba | AAAI 2025 |
| **PhysFormer** | Temporal-Difference Transformer | CVPR 2022 |
| **TSCAN** | Multi-Task Temporal Shift Convolutional Attention Network | NeurIPS 2020 |
| **EfficientPhys**| 2D CNN + Temporal Shift Module | WACV 2023 |
| **PhysNet** | 3D CNN + Encoder-Decoder | BMVC 2019 |

Each model has two variants: one trained on RLAP and the other trained on PURE. They are distinguished by the suffixes `.rlap` and `.pure`, respectively.

## Licensing

The source code and tools in this repository are released under the **MIT License**.

**Important:** Pretrained models and model configurations provided in this repository are derived from academic research. They are the intellectual property of their respective authors and are subject to the license terms specified in their original publications. Please refer to the citations below for details.

## Citation

If you use this script or the included models in your research, cite the relevant papers:

```bibtex
@misc{open-rppg-toolkit,
author = {Kegang Wang},
title = {Open-rPPG: An Open Toolbox for Remote Photoplethysmography},
year = {2024},
howpublished = {\url{https://github.com/KegangWangCCNU/open-rppg}},
note = {GitHub repository, accessed June 2026}
}

@article{physnet,
  title={Remote photoplethysmograph signal measurement from facial videos using spatio-temporal networks},
  author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},
  journal={arXiv preprint arXiv:1905.02419},
  year={2019}
}

@article{mttscan,
  title={Multi-task temporal shift attention networks for on-device contactless vitals measurement},
  author={Liu, Xin and Fromm, Josh and Patel, Shwetak and McDuff, Daniel},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={19400--19411},
  year={2020}
}

@inproceedings{efficientphys,
  title={Efficientphys: Enabling simple, fast and accurate camera-based cardiac measurement},
  author={Liu, Xin and Hill, Brian and Jiang, Ziheng and Patel, Shwetak and McDuff, Daniel},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  pages={5008--5017},
  year={2023}
}

@inproceedings{physformer,
  title={Physformer: Facial video-based physiological measurement with temporal difference transformer},
  author={Yu, Zitong and Shen, Yuming and Shi, Jingang and Zhao, Hengshuang and Torr, Philip HS and Zhao, Guoying},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4186--4196},
  year={2022}
}

@inproceedings{physmamba,
  title={PhysMamba: Efficient Remote Physiological Measurement with SlowFast Temporal Difference Mamba},
  author={Luo, Chaoqi and Xie, Yiping and Yu, Zitong},
  booktitle={Chinese Conference on Biometric Recognition},
  pages={248--259},
  year={2024},
  organization={Springer}
}

@inproceedings{rhythmmamba,
  title={RhythmMamba: Fast, Lightweight, and Accurate Remote Physiological Measurement},
  author={Zou, Bochao and Guo, Zizheng and Hu, Xiaocheng and Ma, Huimin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={10},
  pages={11077--11085},
  year={2025}
}

@article{me-rppg,
  title={Memory-efficient Low-latency Remote Photoplethysmography through Temporal-Spatial State Space Duality},
  author={Wang, Kegang and Tang, Jiankai and Fan, Yuxuan and Ji, Jiatong and Shi, Yuanchun and Wang, Yuntao},
  journal={arXiv preprint arXiv:2504.01774},
  year={2025}
}
```
