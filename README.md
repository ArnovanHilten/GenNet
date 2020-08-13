# **GenNet**
**Framework for Interpretable Neural Networks for genetics**

1. [What is GenNet?](https://github.com/ArnovanHilten/GenNet/#1-what-is-gennet)
1. [Getting started](https://github.com/ArnovanHilten/GenNet/#2-getting-started)
1. [GenNet command line.](https://github.com/ArnovanHilten/GenNet/#3-gennet-command-line)
1. [(optional) Jupyter notebook](https://github.com/ArnovanHilten/GenNet#jupyter-notebook)


## 1. What is GenNet?

<img align = "right" src="https://github.com/ArnovanHilten/GenNet/blob/master/figures/figure1_github.PNG" width="450">
GenNet is a command line tool that can be used to create neural networks for (mainly) genetics. GenNet gives the opportunity to let you decide what should be connected to what. Any information that groups knowledge can therefore be used to define connections in the network. For example, gene annotations can be used to group genetic variants into genes, as seen in the first layer of the image. This creates meaningful and interpretable connections. When the network is trained the network learns which connections are important for the predicted phenotype and assigns these connections a higher weight. For more information about the framework and the interpretation read the paper:

[GenNet framework: interpretable neural networks for phenotype prediction](https://www.biorxiv.org/content/10.1101/2020.06.19.159152v1.full.pdf)

The Gennet framework is based on tensorflow, click [here](https://github.com/ArnovanHilten/GenNet/blob/master/utils/LocallyDirectedConnected_tf2.py) for the custom layer.
</a>
<a name="how"/>

## 2. Getting started

### Prerequisites:

- GenNet uses [CUDA](https://developer.nvidia.com/cuda-10.1-download-archive-base). Please make sure you have the correct version of CUDA installed. GenNet has been tested for:

  * CUDA  9.1 & Tensorflow 1.12.0 
  * CUDA 10.0 & Tensorflow 1.13.1
  * CUDA 10.0 & Tensorflow 2.0.0-beta1
  * CUDA 10.1 & Tensorflow 2.2.0
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

This automatically instals the latest Tensorflow version for which GenNet has been tested. If you have an older version of CUDA install the appriopriate tensorflow-gpu by
`pip install tensorflow-gpu==1.13.1` (change 1.13.1 to your version).

**Activate the environment**
```
source env_GenNet/bin/activate
```

**Install the packages**
```
pip3 install --upgrade pip
pip3 install -r requirements_GenNet.txt

```
*Gennet is ready to use!*

## 3. GenNet command line.
<img align = "right" src="https://github.com/ArnovanHilten/GenNet/blob/master/figures/Gennet_wiki_overview.png?raw=true" width="480">

### Preparing the data
As seen in the overview  the commmand line takes 3 inputs:

1. **genotype.h5** - a genotype matrix, each row is an example (subject) each column is a feature (e.g. genetic variant).
1. **subject.csv** - a .csv file with the following columns:
    * patient_id: am ID for each patient
    * labels: phenotype (with zeros and ones for classification and values for regression)
    * genotype_row: in which row the subject is in the genotype.h5 file
    * set: in which set the patient belongs (1 = training set, 2 =  validation set, 3 = test, others= ignored)
1. **topology** - each row is a "path" of the network, from input to output node.


Topology example (from GenNet/examples/example_study) :

| layer0_node | layer0_name | layer1_node | layer1_name | layer2_node | layer2_name  |
|-------------|-------------|-------------|-------------|-------------|--------------|
| 0           | SNP0        | 0           | HERC2       | 0           | Causal_path  |
| 5           | SNP5        | 1           | BRCA2       | 0           | Causal_path  |
| 76          | SNP76       | 6           | EGFR        | 1           | Control_path |

NOTE: It is important to name the column headers as shown in the table.
The input 5 is connected to the node number 1 in layer 1. That node is connected to node 0 in layer 2. This is the last given layer name so this node is also connected to the output. *The network will have as many layers as there are columns with the name layer.._node*.
Creating 10 columns with the names layer0_node, layer1_node.. layer10_node will results in 10 layers.

Tip: Use as example the example study found in the examples folder.

### Running GenNet

Open the command line and navigate to the GenNet folder. To run the example study run: 
```
python GenNet.py train ./examples/example_study/ 1
```
Choose from: convert, train and plot.

```
python GenNet.py convert --help
python GenNet.py train --help
python GenNet.py plot --help
```

### Jupyter notebook

The orignal jupyter notebooks can be found in the jupyter notebook folder. Navigate to the jupyter notebook folder and start with `jupyter notebook`

### More

[The bioRxiv paper](https://www.biorxiv.org/content/10.1101/2020.06.19.159152v1)

[Run the demo online!](https://tinyurl.com/y8hh8rul)

## Contact
For questions or comments mail to: a.vanhilten@erasmusmc.nl
