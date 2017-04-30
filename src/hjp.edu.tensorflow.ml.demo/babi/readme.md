# MemN2N

Implementation of [End-To-End Memory Networks](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf) with sklearn-like interface using Tensorflow. Tasks are from the [bAbI](https://github.com/facebook/bAbI-tasks)


## Get Started

mkdir ./babi/data/
cd ./babi/data
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar xzvf ./tasks_1-20_v1-2.tar.gz
cd ../
python single.py


## Examples

running a [single bAbI task](./single.py)
running a [joint model on all bAbI tasks](./joint.py)


## Requirements

tensorflow 1.0
scikit-learn 0.17.1
six 1.10.0


## Notes

Single task results are from 10 repeated trails of the single task model accross all 20 tasks with different random initializations.
Joint training results are from 10 repeated trails of the joint model accross all tasks.
