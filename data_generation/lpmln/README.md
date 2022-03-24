# LPMLN
An implementation to compute LPMLN programs using existing ASP and SDD solvers.

## Introduction
LPMLN is a recent addition to probabilistic logic programming languages. Its main idea is to assign weights to stable models in a way similar to Markov Logic assigns weights to models. We present 3 implementations of LPMLN, namely **lpmln-infer** for inference, **lpmln-learn** for weight learning, and **dtlpmln** for DT-LPMLN inference. 

## Try with LPMLN Online
You can try LPMLN without installing it by using our online editor: 
http://reasoning.eas.asu.edu/lpmln-web/ .
A simple user interface allows one to try LPMLN inference, LPMLN learning, and DT-LPMLN (for decision theory) without a hassle. 
Example projects (each consists of an LPMLN program and an evidence file) are also provided for one to start with. 

## Prerequisites
* System: Linux/Ubuntu
* Software: [Anaconda](https://www.anaconda.com/distribution/#linux)
* Python: Python 3.6 or above

## Installation
* [Optional] If your Python version is 3.5 or below, you may create a virtual environment in Anaconda
```
conda create -n env_lpmln python=3.7
conda activate env_lpmln
```
* Install Clingo and gcc
```
conda install -c potassco clingo
sudo apt install gcc
```
* cd to the folder of lpmln package, e.g., (Please modify the path to the lpmln folder accordingly.)
```
cd Downloads/lpmln
```
and execute the following commands
```
python setup.py
sudo ln -s $(pwd)/lpmln_infer.py /usr/local/bin/lpmln-infer
sudo ln -s $(pwd)/lpmln_learn.py /usr/local/bin/lpmln-learn
sudo ln -s $(pwd)/lpmln_dec.py /usr/local/bin/dtlpmln
chmod +x /usr/local/bin/*lpmln*
chmod +x $(pwd)/binSupport/*
```

## Inference Example
Note that, if you created a virtual environment, e.g., **env_lpmln**, when you installed LPMLN, you need to activate this environment before you use the LPMLN package everytime you open a new terminal.
```
conda activate env_lpmln
```

Given the LPMLN program [bird.lpmln](./examples/inference/bird.lpmln), we can find the most probable stable model of it by executing the following command under the folder that contains this .lpmln file.
```
lpmln-infer bird.lpmln
```

## Learning Example
Given the LPMLN program [simple.lpmln](./examples/learn/simple.lpmln) and the observation file [simple.obs](./examples/learn/simple.obs) containing observations (i.e., known facts) in the form of ASP constraints, we can learn the weight of the rules in smoke.lpmln such that the probability of the observations is maximized.  Under the folder that contains this .lpmln file, we use the following command.
```
lpmln-learn simple.lpmln -o simple.obs
```
The learned weight are available in the file **lpmln_learned.weight**.
