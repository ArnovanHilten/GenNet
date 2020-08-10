# HASE
Framework for efficient high-dimensional association analyses.

## Speed test
`run_ExampleStudy.sh` script runs example of association study with **20.000** SNPs, **1000** phenotypes and **1000** subjects.
It runs analysis by chunk of 5000 SNPs (which you can define in `config.py` file). Standard output looks like this:
```
START regression mode...
reading file example_study.csv
There are 1000 ids and 1000 columns 
reading file example_study.csv
There are 1000 ids and 3 columns 
There are 1000 ids
There are 1000 common ids
...
...
...
time to compute GWAS for 1000 phenotypes and 5000 SNPs .... 0.681949138641 sec
Read 15000, processed 15000, total 20000
...
time to compute GWAS for 1000 phenotypes and 5000 SNPs .... 0.565479040146 sec
Read 20000, processed 20000, total 20000
...
experiment finished in 10.0326929092 s
```


## Installation HASE

Navigate to directory where you want to install HASE and clone this repository:
     ```
     git clone https://github.com/roshchupkin/hase.git
     ```
## Update HASE 

You can update HASE to newest version using `git`. Navigate to your HASE folder (where you cloned git repository):    
     ```
     git pull
     ```

## Installation requirements

Your system might already satisfied requirements, we suggest first try to run test example from Testing header below. 

1. HDF5 software (python packages `tables` and `h5py` require this installation). If it is not installed on you system, 
you can download to your home directory the latest source code [hdf5](https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.16/src/).
    ```
    tar -xf ~/hdf5-1.8.16.tar.gz
    cd ~/hdf5-1.8.16/
    ./configure 
    make 
    make install
    ```
    Then you need to add one line to your `.bachrc` or `.bash_profile` file in your home directory.

    ```
    export HDF5_DIR=~/hdf5-1.8.16/hdf5/
    ```

2. BLAS and LAPACK linear algebra libraries for `scipy` and `numpy`. 
     ```
     sudo apt-get install gfortran libopenblas-dev liblapack-dev
     ```
    If this does not work or raise errors, then you might need to follow instruction from [scipy](http://www.scipy.org/scipylib/building/index.html) website. 

3. You need to install python. You can download python from official website [python](https://www.python.org/) 
or install one of the python distribution for scientific research, such as [Anaconda](https://store.continuum.io/cshop/anaconda/),
[Enthought Canopy](https://www.enthought.com/products/canopy/) or [Python(x,y)](http://python-xy.github.io/).
And then you need to install (or first uninstall) `scipy` and `numpy` python libraries.
     ```
     pip install scipy 
     pip install numpy
     ```
 
    To check linkage in numpy:
      ```
      python
      >>> import numpy as np
      >>> np.__config__.show()
      ```
  
    And you should see something like this:
  
      ```
      lapack_opt_info:
        libraries = ['openblas', 'openblas']
        library_dirs = ['/cm/shared/apps/openblas/0.2.9-rc2/lib']
        language = f77
    blas_opt_info:
        libraries = ['openblas', 'openblas']
        library_dirs = ['/cm/shared/apps/openblas/0.2.9-rc2/lib']
        language = f77
    openblas_info:
        libraries = ['openblas', 'openblas']
        library_dirs = ['/cm/shared/apps/openblas/0.2.9-rc2/lib']
        language = f77
    blas_mkl_info:
      NOT AVAILABLE
      ```  
     
4. Install python packages listed in `requirements.txt` file. (you can use package manager which comes with your python `pip` or `conda` to install packages):
    * bitarray
    * argparse
    * cython
    * matplotlib
    * scipy
    * numpy
    * pandas
    * h5py
    * tables

## Testing

1. Navigate to HASE directory and type `python hase.py -h`, you should see help message.
2. Navigate to HASE directory and type `sh run_ExampleStudy.sh`, it should start running toy example of high-dimensional GWAS.
 
## User Guide
[wiki](https://github.com/roshchupkin/hase/wiki).
## Requirements
1. HDF5 software.
2. BLAS and LAPACK linear algebra libraries.  
3. Python. 
4. Python packages:
    * bitarray
    * argparse
    * cython
    * matplotlib
    * scipy
    * numpy
    * pandas
    * h5py
    * tables
5. [Git](https://git-scm.com/).


## Citation 
If you use HASE framework, please cite:

[Roshchupkin, G. V. et al. HASE: Framework for efficient high-dimensional association analyses. Sci. Rep. 6, 36076; doi: 10.1038/srep36076 (2016)](http://www.nature.com/articles/srep36076) 

## Licence
This project is licensed under GNU GPL v3.

## Authors
Gennady V. Roshchupkin (Department of Epidemiology, Radiology and Medical Informatics, Erasmus MC, Rotterdam, Netherlands)

Hieab H. Adams (Department of Epidemiology, Erasmus MC, Rotterdam, Netherlands) 

## Contacts

If you have any questions/suggestions/comments or problems do not hesitate to contact us!

* g.roshchupkin@erasmusmc.nl
* h.adams@erasmusmc.nl
 
