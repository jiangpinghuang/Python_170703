# MemN2N

Implementation of [End-To-End Memory Networks](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf) with tensorflow for language model. The training data is penn treebank (ptb).


## Get Started

To train a model with 6 hops and memory size of 100, run the following command:
	$ python main.py --nhop 6 --mem_size 100


## Requirements

tensorflow 1.0
future
progress


## Notes

The code from http://carpedm20.github.io/.
