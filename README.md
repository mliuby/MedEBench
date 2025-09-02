# [EMNLP2025 Findings] MedEBench: Diagnosing Reliability in Text-Guided Medical Image Editing

[![arXiv](https://img.shields.io/badge/arXiv-2506.01921-b31b1b.svg)](https://arxiv.org/abs/2506.01921)

**MedEBench** is a benchmark for evaluating text-instructed image editing methods in the **medical domain**, spanning diverse anatomy and surgical editing tasks.

This repo contains:
- The full dataset used in the paper
- Scripts for running and evaluating editing models
- Baselines and prompt templates

👉 **[Read the paper on arXiv](https://arxiv.org/abs/2506.01921)**

---

## 📦 Dataset

- The dataset is included under `editing/`
- You can also download it from:  
  👉 https://huggingface.co/datasets/LIUMinghao/MedEBench

It includes:
- `editing_metadata.json`: Prompts, image paths, task types
- `previous/`, `changed/`, `previous_mask/`: Input, target, and mask images

---

## 🗂️ Repository Structure

```
MedEBench/
├── GSA_requirements.txt
├── LICENSE
├── requirements.txt
├── structure.txt
├── editing/
│   ├── changed/
│   ├── editing_metadata.json
│   ├── previous/
│   └── previous_mask/
├── src/
│   ├── GSA_requirements.txt
│   ├── attention_map.ipynb
│   ├── baseline.py
│   ├── batch_accelerate.py
│   ├── gemini_edit.py
│   ├── imagic_edit.py
│   ├── dataset_construction/
│   └── eval_metrics/
```

---

## 📋 Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

If using the GSA modules, also install:

```bash
pip install -r src/GSA_requirements.txt
```

---

## 📖 Citation

If you use this project or dataset, please cite us:

```bibtex
@misc{liu2025medebenchrevisitingtextinstructedimage, 
  title={MedEBench: Diagnosing Reliability in Text-Guided Medical Image Editing}, 
  author={Minghao Liu and Zhitao He and Zhiyuan Fan and Qingyun Wang and Yi R. Fung},
  year={2025},
  eprint={2506.01921},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.01921}
}
```

---

## 📬 Contact

For questions or contributions, feel free to open an issue or contact the authors via the email listed in the [paper](https://arxiv.org/abs/2506.01921).
