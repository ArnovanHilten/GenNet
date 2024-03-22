[![DOI](https://zenodo.org/badge/240289809.svg)](https://zenodo.org/badge/latestdoi/240289809)
![Build](https://github.com/ArnovanHilten/GenNet/actions/workflows/tests.yml/badge.svg)

# **GenNet**
**Framework for Interpretable Neural Networks for Genetics**

1. [What is GenNet?](https://github.com/ArnovanHilten/GenNet/#1-what-is-gennet)
1. [Getting started](https://github.com/ArnovanHilten/GenNet/#2-getting-started)
1. [GenNet command line.](https://github.com/ArnovanHilten/GenNet/#3-gennet-command-line)
1. [Wiki](https://github.com/ArnovanHilten/GenNet/wiki)


## 1. What is GenNet?

<img align = "right" src="https://github.com/ArnovanHilten/GenNet/blob/master/figures/figure1_github.PNG" width="450">
GenNet is a command line tool that can be used to create neural networks for (mainly) genetics. GenNet gives the opportunity to let you decide what should be connected to what. Any information that groups knowledge can therefore be used to define connections in the network. For example, gene annotations can be used to group genetic variants into genes, as seen in the first layer of the image. This creates meaningful and interpretable connections. When the network is trained the network learns which connections are important for the predicted phenotype and assigns these connections a higher weight. For more information about the framework and the interpretation read the paper:

[GenNet framework: interpretable neural networks for phenotype prediction](https://www.nature.com/articles/s42003-021-02622-z.pdf)

The Gennet framework is based on tensorflow, click [here](GenNet_utils/LocallyDirected1D.py) for the custom layer.
</a>
<a name="how"/>

## 2. Getting started

Follow the instructions below to get started.

> [!TIP]
> Check the [A to Z Colab tutorial](https://colab.research.google.com/github/ArnovanHilten/GenNet/blob/master/examples/A_to_Z/GenNet_A_to_Z.ipynb) for an overview on how to use GenNet with your own data!

### Prerequisites:

- GenNet can use CPU or GPU (which can be quite a bit faster for deeper networks) If you want to use cuda, please make sure you have the correct version of CUDA installed [CUDA](https://developer.nvidia.com/cuda-10.1-download-archive-base). GenNet has been tested for:

  * Python 3.5,  CUDA  9.1,  Tensorflow 1.12.0 
  * Python 3.5,  CUDA 10.0,  Tensorflow 1.13.1
  * Python 3.5,  CUDA 10.0,  Tensorflow 2.0.0-beta1 
  * Python 3.6-3.7,  CUDA 10.1,  Tensorflow 2.2.0 (currently default and recommended)
  * Python 3.* Tensorflow 2.2 to 2.5 CPU
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

This automatically installs the latest Tensorflow version for which GenNet has been tested. If you have an older version of CUDA install the appriopriate tensorflow-gpu by
`pip install tensorflow-gpu==1.13.1` (change 1.13.1 to your version).

**Activate the environment**
```
source ~/env_GenNet/bin/activate
```

**Install the packages**
```
pip3 install --upgrade pip
pip3 install -r requirements_GenNet.txt

```
*GenNet is ready to use!*

Navigate to the GenNet folder and use the following command to run the example:
```
python GenNet.py train -path ./examples/example_classification/ -ID 1
```

Check the [wiki](https://github.com/ArnovanHilten/GenNet/wiki) for more info!

## 3. GenNet command line.
<img align = "right" src="https://github.com/ArnovanHilten/GenNet/blob/master/figures/Gennet_wiki_overview.png?raw=true" width="480">

### Preparing the data
*NOTE: In python indices start from zero*

As seen in the overview the commmand line takes 3 inputs:

1. **genotype.h5** - a genotype matrix, each row is a sample/subject/patient, each column is a feature (i.e. genetic variant). The genotype file can be automatically generated from **plink** files and **VCF** files using `python GenNet.py convert`, use `python GenNet.py convert --help` for more options or check [HASE wiki convert](https://github.com/roshchupkin/hase/wiki/Converting-Data)
1. **subject.csv** - a .csv file with the following columns:
    * patient_id: am ID for each patient 
    * labels: phenotype (with zeros and ones for classification and continuous values for regression)
    * genotype_row: The row in which the subject can be found in the genotype matrix (genotype.h5 file)
    * set: in which set the subject belongs (1 = training set, 2 =  validation set, 3 = test, others= ignored)
1. **topology** - This file describes the whole network: each row should be a "path" of the network, from input to output node. This file defines thus each connections in the network, giving you the freedom to design your network the way you want. In the GenNet framework we used biological knowledge such as gene annotations to do define meaningful connections, we included some helper functions to generate a topology file using Annovar. See the topoogy help for more information: `python GenNet.py topology --help`




Topology example:

| layer0_node | layer0_name | layer1_node | layer1_name | layer2_node | layer2_name  |
|-------------|-------------|-------------|-------------|-------------|--------------|
| 0           | rs916977    | 0           | HERC2       | 0           | Ubiquitin mediated proteolysis|
| 1           | rs766173    | 1           | BRCA2       | 1           | Breast cancer  |
| 5           | rs1799944   | 1           | BRCA2       | 1           | Breast cancer  |
| 6           | rs4987047   | 1           | BRCA2       | 1           | Breast cancer  |
| 1276        | SNP1276     | 612         | UHMK1       | 2          | Tyrosine metabolism |

NOTE: It is important to name the column headers as shown in the table.

The first genetic variant in the genotypefile (row number zero!), named rs916977,  is connected to the HERC2 node in the first layer. The HERC2 gene is node number zero. This node is conncted to the 'Ubiquitin mediated proteolysis' pathway which is the first node in the following layer. The next node is the end node which should not be included.

The second genetic variant 'rs766173' is connected to BRCA2 (node number 1 in the first layer), followed by the breast cancer pathway (node number 1 in the layer2), folowed by the end node.

The sixth(!) genetic variant 'rs1799944' is also connected to BRCA2 (whic was node number 1 in the first layer), followed by the breast cancer pathway (again node number 1 in the layer2), folowed by the end node.

All rows together describe all the connections in the network. Each layer should be described by a column layer#_node and a column layer#_name with # denoting the layer number.   

Tip: Check the topology files in the examples folder.

### Running GenNet

Open the command line and navigate to the GenNet folder. Start training by:
```
python GenNet.py train -path {/path/to/your/folder} -ID {experimment number}
```
For example:
```
python GenNet.py train -path ./examples/example_classification/ -ID 1
```
or
```
python GenNet.py train -path ./examples/example_regression/ -ID 2 -problem_type regression
```
Choose from: convert, topology, train, plot and interpret. For the options check the [wiki](https://github.com/ArnovanHilten/GenNet/wiki) or use:

```
python GenNet.py convert --help
python GenNet.py train --help
python GenNet.py plot --help
python GenNet.py topology --help
python GenNet.py interpret --help
```

#### GenNet output

After training your network it saved together with its results. Results include a text file with the performance, a .CSV file with all the connections and their weights, a .h5 with the best weights on the validation set and a plot of the training and validation loss. Using these files we can create visualizations to better understand the network.

The .CSV file with the weights can be used to create your own plot but `python GenNet.py plot` also has standard plots available. First we calculate the relative importance by multiplying all the weights between the output and each input. This can then be used to see the importance of each gene:

##### Manhattan plot
<img align = "center" src="https://github.com/ArnovanHilten/GenNet/blob/master/figures/example_manhattan.png">

Or can be used in a Sunburt plot to get an overview of the whole network!

##### Sunburst plot
<img align = "center" src="https://github.com/ArnovanHilten/GenNet/blob/master/figures/Sunburst_pathway_schizophrenia.png">




### GenNet Interpretation module

GenNet offers a number of intepretation methods, to find important features and interacting features.

- get_weight_scores: uses the weights to calculate the importance of each feature and node
- DeepExplain: uses the gradient (see DeepExplain) to calculate the importance
- RLIPP: uses logistic regression with signals to and from the node to calculate a measure of non-linearity for all nodes
- NID: Finds interacting features based on the features with the strongest weights
- DFIM: perturbs each input (or N inputs in the order of importance), and tracks which other features change importance to find interacting features
- PathExplain: Uses the Expected Hessian to find interacting features

For more information use: `python GenNet.py interpret --help`


### Jupyter notebook

The original jupyter notebooks can be found in the jupyter notebook folder. Navigate to the jupyter notebook folder and start with `jupyter notebook`. The notebooks are not updated but can be a useful source to understand the main code and/or to create .npz masks (to define connections between layers). For more freedom in designing your own networks you can define your network [here](https://github.com/ArnovanHilten/GenNet/blob/master/GenNet_utils/Create_network.py#L25) and create masks using the notebooks.

### More

[A to Z tutorial in Google Colabs, try GenNet with a single click!](https://colab.research.google.com/github/ArnovanHilten/GenNet/blob/master/examples/A_to_Z/GenNet_A_to_Z.ipynb)

[GenNet is also available on Superbio.ai!](https://app.superbio.ai/apps/136/)

[Paper](https://www.nature.com/articles/s42003-021-02622-z)

[The bioRxiv paper](https://www.biorxiv.org/content/10.1101/2020.06.19.159152v2.full.pdf)

[All plots](https://github.com/ArnovanHilten/GenNet_paper_plots)

[Trained networks](https://github.com/ArnovanHilten/GenNet_Zoo)

[Toy with the demo with the basic principles online!](https://tinyurl.com/y8hh8rul)

[(Deprecated) Jupyter notebooks](https://github.com/ArnovanHilten/GenNet#jupyter-notebook)



## Contact
For questions or comments make an issue or mail to: a.vanhilten@erasmusmc.nl
