# <div align="center">Instance-aware Model Ensemble With Distillation For Unsupervised Domain Adaptation</div>

## Introduction 
[[`Paper`](http://arxiv.org/abs/2211.08106)] 
<div align="center">
  <img width="100%" alt="Instance-aware Model Ensemble With Distillation" src="https://github.com/wuweimin23/IMED/blob/master/fig/1.png">
</div>

<p align="justify">The linear ensemble-based strategy (i.e., averaging ensemble) has been proposed to improve the performance in unsupervised domain adaptation (UDA) task. However, a typical UDA task is usually challenged by dynamically changing factors, such as variable weather, views and background in the unlabeled target domain. Most previous ensemble strategies ignore UDA’s dynamic and uncontrollable challenge, facing limited feature representations and performance bottlenecks. To enhance the
model adaptability between domains and reduce the computational cost when deploying the ensemble model, we propose a novel framework, namely Instance-aware Model Ensemble With Distillation (IMED), which fuses multiple UDA component models adaptively according to different instances and distills these components into a small model. The core idea of IMED is a dynamic instance-aware ensemble strategy, where for each instance, a non-linear fusion sub-network is learned that fuses the extracted features and predicted labels of multiple component models. The non-linear fusion method can help the ensemble model handle dynamically changing factors. After learning a large-capacity ensemble model with good adaptability to different changing factors, we leverage the ensemble teacher model to guide the learning of a compact student model by knowledge distillation. </p>

## Getting Started

The code about the main experiments and ablation experiments can been seen in 'examples'
The sepcific running scripts for each experiment are described in three files:
run_office31.sh : experiments on dataset Office31
run_office_home.sh : experiments on dataset Office-Home
run_visda_2017.sh : experiments on dataset Visda-2017


## Training

Also, we have examples in the directory `examples`. A typical usage is 
```shell script
# Train a CDANs on Office-31 Amazon -> Webcam task using ResNet 50.
# Assume you have put the datasets under the path `data/office-31`, 
# or you are glad to download the datasets automatically from the Internet to this path
python dann.py data/office31 -d Office31 -s A -t W -a resnet50  --epochs 20
```

In the directory `examples`, you can find all the necessary running scripts to reproduce the benchmarks with specified hyper-parameters.

##Results

We following table reports the results of our method compared to the state-of-the-art results.

<div align="center">
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Office-31</th>
            <th>Office-Home</th>
            <th>Visda-2017</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=12> SOTA</td>
            <td>94.1%</td>
            <td>89.8%</td>
            <td>84.3%</td>
        </tr>
        <tr>
            <td rowspan=12> IMED</td>
            <td>94.4%</td>
            <td>89.9%</td>
            <td>85.1%</td>
        </tr>
        </tbody>
</table>
</div>

## Acknowledgement
Our implementation is based on the [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library). 
