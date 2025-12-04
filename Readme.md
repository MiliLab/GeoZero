
<div align="center">

<h1>GeoZero: Incentivizing Reasoning from Scratch on Geospatial Scenes</h1>


Di Wang<sup>1</sup>, 
Shunyu Liu<sup>2</sup>, 
Wentao Jiang<sup>1</sup>, 
Fengxiang Wang<sup>3</sup>, 
Yi Liu<sup>1</sup>, 
Xiaolei Qin<sup>1</sup>, 
Zhiming Luo<sup>1</sup>, 

Chaoyang Zhou<sup>1</sup>, 
Haonan Guo<sup>1</sup>, 
Jing Zhang<sup>1 †</sup>, 
Bo Du<sup>1 †</sup>, 
Dacheng Tao<sup>2</sup>, 
Liangpei Zhang<sup>1 †</sup>  

<sup>1</sup> Wuhan University,  <sup>2</sup> Nanyang Technological University,  <sup>3</sup> Shanghai AI Laboratory.


<sup>†</sup> Corresponding author

</div>


<p align="center">
  <a href="#-update">Update</a> |
  <a href="#-overview">Overview</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-models">Models</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-statement">Statement</a>
</p >



## 🔥 Update


**2025.12.04**

- All components required for building an inference demo have been prepared.  
- The updated **model weights** are available on:  
  - **[Hugging Face](https://huggingface.co/hjvsl/GeoZero/tree/main/GeoZero-8B-without-RFT)**  
  - **[Baidu Drive](https://pan.baidu.com/s/1nJjBwO4UlVv4GFl60gjM3w?pwd=15gn)**  
- The **JSON annotation files for the test sets of several benchmarks** used in our evaluation have been released and are available at:  
  - **[Hugging Face](https://huggingface.co/datasets/hjvsl/GeoZero_Eval_Datasets/tree/main)**  
  *(Note: Only the JSON files are provided; the corresponding images must be downloaded from the original datasets.)*


**2025.12.01**
- The paper is post on arXiv! **([arXiv](https://arxiv.org/abs/2511.22645))** 


## 🌞 Overview

We present GeoZero, the first MLLM capable of performing emergent reasoning on geospatial scenes from scratch without any predefined CoT supervision. To encourage deep and reliable reasoning while maintaining answer accuracy, we construct two datasets, GeoZero-Instruct and GeoZero-Hard. GeoZero-Instruct allows the model to acquire preliminary geospatial knowledge through supervised fine-tuning, while GeoZero-Hard stimulates deep reasoning during the subsequent reinforcement learning stage. We also propose Answer-Anchored Group Relative Policy Optimization (A$^2$GRPO), where the reasoning process is regularized by the model’s own answers, encouraging diverse yet accurate thinking. GeoZero not only reduces annotation costs but also enhances the cognitive capability of MLLMs, offering new insights toward general geospatial AI.


<figure>
<div align="center">
<img src=Figs/framework.png width="100%">
</div>

<div align='center'>
 
**Figure 1. Framework of GeoZero.**

</div>
<br>

## 📖 Datasets

GeoZero relies on multiple remote sensing benchmarks for both model development and evaluation. Please manually download the corresponding image datasets from their original sources. 

### 🔗 Recommended Data Sources

| Dataset | Dataset | Dataset |
|--------|---------|---------|
| [VHM-Instruct](https://github.com/opendatalab/VHM/blob/main/docs/Data.md) | [RESISC-45](https://gcheng-nwpu.github.io/#Datasets) | [EuroSAT](https://github.com/phelber/EuroSAT) |
| [AID](https://captain-whu.github.io/AID/) | [NASC-TG2](https://aistudio.baidu.com/datasetdetail/86451) | [fMoW](https://github.com/fMoW/dataset) |
| [WHU-RS19](https://captain-whu.github.io/BED4RS/) | [RSVQA](https://rsvqa.sylvainlobry.com/) | [UCM](http://weegee.vision.ucmerced.edu/datasets/landuse.html) |
| [RSVG](https://drive.google.com/file/d/1kgnmVC6FVKdxCwaoG77sOfkaIHS_XiFt/view) | [DIOR-RSVG](https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_) | [SkyEye-968k](https://huggingface.co/datasets/ZhanYang-nwpu/SkyEye-968k) |
| [VRSBench](https://huggingface.co/datasets/xiang709/VRSBench/tree/main) | [SIRI-WHU](https://rsidea.whu.edu.cn/code_data/USGS_Dataset/The%20USGS%20image%20dataset%20of%20SIRI-WHU%20annotation.zip) | [UCM-Captions](https://github.com/201528014227051/RSICD_optimal?tab=readme-ov-file) |
| [Sydney-Captions](https://github.com/201528014227051/RSICD_optimal?tab=readme-ov-file) | [NWPU-Captions](https://github.com/HaiyanHuang98/NWPU-Captions) | [RSICD](https://github.com/201528014227051/RSICD_optimal?tab=readme-ov-file) |

We provide **pre-formatted JSON annotation files** to ensure consistent data loading and usage:

### Training data

Coming Soon.

### Evaluation data

Evaluation samples across different benchmarks are available on our continually updated Hugging Face dataset repository:

👉 [GeoZero_Eval_Datasets](https://huggingface.co/datasets/hjvsl/GeoZero_Eval_Datasets/tree/main)

## 🚀 Models

|Model | Weights |
| :------- | :------: |
|GeoZero w/o RFT| [Hugging Face](https://huggingface.co/hjvsl/GeoZero/tree/main/GeoZero-8B-without-RFT) & [Baidu Drive](https://pan.baidu.com/s/1nJjBwO4UlVv4GFl60gjM3w?pwd=15gn) |


## 🔨 Usage

### Training

Wait for update.

### Inference

We provide an inference script for Qwen3-VL and related models on various remote sensing vision–language tasks:

```
python single_infer_eval_qwen3vl_think.py \
--model_path [model path] \
--json_path [dataset json path] \
--output_path [output saved path] \
--task [task type] --batchsize 4 --gpu [gpu id] --system [whether use the system prompt (Type1)]
```

## 🍭 Results


<div align="center">
<img src=Figs/result.png width="100%">
</div>


## ⭐ Citation

If you find GeoZero helpful, please give a ⭐ and cite it as follows:

```
@article{wang2025geozero,
  title   = {GeoZero: Incentivizing Reasoning from Scratch on Geospatial Scenes},
  author  = {Wang, Di and Liu, Shunyu and Jiang, Wentao and Wang, Fengxiang and Liu, Yi and Qin, Xiaolei and Luo, Zhiming and Zhou, Chaoyang and Guo, Haonan and Zhang, Jing and Du, Bo and Tao, Dacheng and Zhang, Liangpei},
  journal = {arXiv preprint arXiv:2511.22645},
  year    = {2025}
}
```

## 🎺 Statement

For any other questions please contact di.wang at [gmail.com](mailto:wd74108520@gmail.com) or [whu.edu.cn](mailto:d_wang@whu.edu.cn).


## 💖 Thanks
This project is based on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [ms-swift](https://github.com/modelscope/ms-swift), [RSEvalKit](https://github.com/fitzpchao/RSEvalKit),  Thanks for their wonderful work!<br>
