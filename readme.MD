# CIS 700 Project
This is the repository for course CIS 700 Deeplearning Theorem Proving's term project. We trained a GGNN using task 15 of bAbI datast. 

## Team members
* Ruipeng LIU (rliu02@syr.edu)
* Jiajing Chen (jchen152@syr.edu)
* Minmin Yang (myang47@syr.edu)
* Chenbin Pan (cpan14@syr.edu) 

## Dependencies
* pytorch
* tensorboard
* PyYAML
* numpy
* tqdm 

If you want to use gpu to accelerate the training process, then please make sure that cuda is installed prior applying option `--device cuda` to the execution command.
## Usage
To run the program, simply execute the following command:
```commandline
python main.py
```
Several options can be applied to this command. The options are list as following:

* `--exp_name`: Name of the experiment. A directory within `Experiment` will be created with this name to record the training logs. Default: `default`.
* `--epochs`: Number of training epochs. Default: `50`.
* `--train`: Flag for training. Default `True`
* `--data_path`: Path for the data. Default: `babi_data/processed_1/train/15_graphs.txt`
  * Depend on the OS that you are using, this option may need corresponding adjustment.
* `--batch_size`: Batch size. Default: `15`
* `--lr`: Learning rate. Default: `0.01`
* `--device`: Device used for training model. Default: `cuda`
  * Again, if this option is set to `cuda`, please make sure that cuda is installed.
* `--opt`: Optimizer used in training. Options:`SGD, Adam`. Default: `Adam`
* `--state_dim`: Dimension for states. Default: `4`
* `--annotation_dim`: Annotation dimension. Default: `1`
* `--edge_type`: Types of edges in the graph. Default: `2`
* `--n_nodes`: Default: `8`
* `--n_step`: Number of time that propagation is done. Default: `5`\

For viewing training curve after training, please execute this command in the project's root directory.
```commandline
tensorboard --logdir Experiment
```
## Acknowledgement 
The model and data used in this repository is based on [this](https://github.com/chingyaoc/ggnn.pytorch) repository.\
The original bAbI project can be found [here](https://github.com/facebookarchive/bAbI-tasks).



