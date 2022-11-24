Reference
"https://github.com/thuml/Transfer-Learning-Library"

This is the source code for paper 
'Instance-aware Model Ensemble With Distillation For Unsupervised Domain Adaptation'
arXiv link: http://arxiv.org/abs/2211.08106

![Alt text](https://github.com/wuweimin23/IMED/blob/master/fig/1.png)

## Intorduction

The code about the main experiments and ablation experiments can been seen in 'examples'
The sepcific running scripts for each experiment are described in three files:
run_office31.sh : experiments on dataset Office31
run_office_home.sh : experiments on dataset Office-Home
run_visda_2017.sh : experiments on dataset Visda-2017


## Documentation

Also, we have examples in the directory `examples`. A typical usage is 
```shell script
# Train a CDANs on Office-31 Amazon -> Webcam task using ResNet 50.
# Assume you have put the datasets under the path `data/office-31`, 
# or you are glad to download the datasets automatically from the Internet to this path
python dann.py data/office31 -d Office31 -s A -t W -a resnet50  --epochs 20
```

In the directory `examples`, you can find all the necessary running scripts to reproduce the benchmarks with specified hyper-parameters.

