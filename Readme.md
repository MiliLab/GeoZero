
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
  <a href="#-pretrained-models">Models</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-statement">Statement</a>
</p >

<figure>
<div align="center">
<img src=Fig/logo1.png width="20%">
</div>
</figure>





## 🔥 Update

**2025.11.25**
- The paper is post on arXiv! **([arXiv 2406.11519](https://arxiv.org/abs/2406.11519))** 


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

Coming Soon.

## 🚀 Models

Coming Soon.



## 🔨 Usage

### Inference

We pretrain the HyperSIGMA with SLURM. This is an example of pretraining the large version of Spatial ViT:

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

If you find our HyperSIGMA helpful, please give a ⭐ and cite it as follows:

```

```

## 🎺 Statement

For any other questions please contact di.wang at [gmail.com](mailto:wd74108520@gmail.com) or [whu.edu.cn](mailto:d_wang@whu.edu.cn).


## 💖 Thanks
This project is based on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [ms-swift](https://github.com/modelscope/ms-swift),  Thanks for their wonderful work!<br>
