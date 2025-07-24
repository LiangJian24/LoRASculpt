# [CVPR'25 Oral] LoRASculpt

This repository is built for the paper [LoRASculpt: Sculpting LoRA for Harmonizing General and Specialized Knowledge in Multimodal Large Language Models](https://arxiv.org/abs/2503.16843).

<div align="center">
<img alt="method" src="images/LoRASculpt.png">
</div>

## News
* [2025-05] Repo created. Code will be released soon.
* [2025-07] Code released.


## Preparation
1. Clone this repository and navigate to LoRASculpt folder:
    ```bash
    git clone https://github.com/LiangJian24/LoRASculpt
    cd LoRASculpt
    ```

2. Install package:
    ```Shell
    conda create -n lorasculpt python=3.10 -y
    conda activate lorasculpt
    pip install --upgrade pip
    pip install -e .
    ```

3. Install additional packages for training cases:
    ```
    pip install -e ".[train]"
    pip install flash-attn --no-build-isolation
    ```

4. Download the required datasets and place them in the corresponding folder.


## Usage
1. Set the correct paths in the scripts under `./scripts/v1_5`.

2. Run the following training script to train on downstream task:
    ```bash
    bash ./scripts/v1_5/train/ours-train.sh
    ```

3. Run the following script to evaluate upstream and downstream performance:
   ```bash
   bash ./scripts/v1_5/eval/eval_all.sh
   ```


## Citation
If you find LoRASculpt useful for your research and applications, please cite using this BibTeX:
```bibtex
@InProceedings{Liang_2025_CVPR,
    author    = {Liang, Jian and Huang, Wenke and Wan, Guancheng and Yang, Qu and Ye, Mang},
    title     = {LoRASculpt: Sculpting LoRA for Harmonizing General and Specialized Knowledge in Multimodal Large Language Models},
    booktitle = {CVPR},
    year      = {2025}
}
```

## Acknowledgement
Our repo is built on [LLaVA](https://github.com/haotian-liu/LLaVA). We thank the authors for sharing their code.


## Related Projects
Keeping Yourself is Important in Downstream Tuning Multimodal Large Language Model
[[Paper](https://arxiv.org/abs/2503.04543)][[Project Page](https://github.com/WenkeHuang/Awesome-MLLM-Tuning)]
