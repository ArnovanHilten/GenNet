# GenNet
## Framework for Interpretable Neural Networks
[The bioRxiv paper](https://www.biorxiv.org/content/10.1101/2020.06.19.159152v1)

[Run the demo online!](https://tinyurl.com/y8hh8rul)

## Principle
GenNet comes with a layer in Tensorlfow that lets you decide what to connect to what. Any information that groups knowledge can therefore be used to define interpretable connections. For example, gene annotations are used to group genetic variants into genes in the first layer. This creates meaningful and interpretable connections, the learned weights represent the importance of the connections. 

<img src="https://github.com/ArnovanHilten/GenNet/blob/master/figures/figure1_github.PNG" width="500">


## Prerequisites:


- GenNet has been tested with an NVIDIA GPU with:

  * Cuda  9.1 & Tensorflow 1.12.0 
  * Cuda 10.0 & Tensorflow 1.13.1
  * Cuda 10.0 & Tensorflow 2.0.0-beta1
  * Cuda 10.1 & Tensorflow 2.2.0

- [HASE](https://github.com/roshchupkin/hase) is used for the conversion of the data to .h5 for parallel reading and writing.

- [Annovar](https://doc-openbio.readthedocs.io/projects/annovar/en/latest/) for obtaining gene annotations

## Get started

### Clone the repository

Open terminal. Navigate to the a place where you want to store the project. Clone the repository:
```
git clone https://github.com/arnovanhilten/GenNet
```
### Install the virtual envionment

**Navigate to the home folder and create a virtual environment**
```
cd ~
python3 -m venv env_GenNet
```

**Activate the environment**
```
source env_GenNet/bin/activate
```

**Install the packages**
```
pip3 install --upgrade pip
pip3 install -r requirements_GenNet.txt
```
**Start jupyter notebook:**
```
jupyter notebook
```
## Contact
For questions or comments mail to: a.vanhilten@erasmusmc.nl
