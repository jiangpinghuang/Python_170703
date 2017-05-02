## End-to-End Memory Networks in TensorFlow for Language Model

End-to-End Memory Networks implementation in TensorFlow for language model on the Penn Treebank (ptb) data.


### Dependencies

Python 2.7.x

TensorFlow 1.0.1

Numpy 1.11.2

progress 1.3

future 0.16.0


### Quickstart

The ptb data is from:

* [Wojciech Zaremba's lstm repo](https://github.com/wojzaremba/lstm)

and put the 'data' folder into 'lang' folder.
	
Use 'main.py' for running a model with 2 hops and memory size of 20, run the following command:

	$ python main.py --nhop 2 --mem_size 20
	
To see all training options, run the following command:

	$ python main.py --help
	
	
### Reference

The memory model is from:
* [Memory Networks](https://arxiv.org/pdf/1410.3916.pdf). Weston et al., ICLR 2016.
* [End-to-End Memory Networks](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf). Sukhbaatar et al., NIPS 2015.


### Acknowledgments

Our implementation utilizes code from the following:
* [Taehoon Kim's MemN2N-tensorflow repo](https://github.com/carpedm20/MemN2N-tensorflow)


### License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

