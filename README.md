# SMCDC2021

Repository for our submission to the Smoky Mountains Computational Sciences and Engineering Conference Data Challenge 2021 - Finding Novel Links in COVID-19 Knowledge
Graph.

## Setup
Note that this repository was created in `Python 3.7.10` and all GCN models were trained on a NVIDIA TITAN V GPU. To set up, perform the following:

1. `git clone https://github.com/RemingtonKim/SMCDC2021.git`
2. `cd SMCDC2021`
3. `pip install -r requirements.txt`
4. Download necessary data from [here](https://github.com/ORNL/smcdc-2021-covid-kg) and store in `./data`.

## Code Description
The following describes the purpose of the files in `./src`. 

File|Description
---|---
`preprocessing.ipynb`| Cleans raw data and preprocesses it into networkX and PyTorch compatible formats.
`analysis.ipynb` | Performs various network analyses.
`baselines.ipynb` | Runs all link and leadtime prediction heuristic and node2vec baselines.
`models.py` | Contains all the GCN models built in PyTorch.
`utils.py` | Contains helper functions.
`train_*_*.py` | Performs training, validation, and testing for a GCN model. 

Note that `preprocessing.ipynb` should be run before anything else as its outputs are used in the other notebooks and files. To run a GCN model, run `python3 train_*_*.py`. Set `MODE = 'test'` if the model has been trained. `.pth` and log files for trained models will show up in `./models`. 

## Contact
* Remington Kim (remingtonskim@gmail.com)
* Yue Ning (yue.ning@stevens.edu)