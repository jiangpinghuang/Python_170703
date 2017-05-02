## End-to-End Memory Networks in TensorFlow for bAbI data

End-to-End Memory Networks implementation in TensorFlow for question answering on the bAbI data.


### Dependencies

Python 2.7.x

TensorFlow 1.0.1

Numpy 1.11.2

scikit-learn 0.18.1

six 1.10.0


### Quickstart

Run the included shell script to fetch the data

	bash data.sh
	
Use 'single.py' for running a [single bAbI task].

	$ python single.py
	
Use 'joint.py' for running a [joint model on all bAbI tasks].

	$ python joint.py
	
	
### Reference

The memory model is from:
* [Memory Networks](https://arxiv.org/pdf/1410.3916.pdf). Weston et al., ICLR 2016.
* [End-to-End Memory Networks](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf). Sukhbaatar et al., NIPS 2015.


### Acknowledgments

Our implementation utilizes code from the following:
* [Dominique Luna's memn2n repo](https://github.com/domluna/memn2n)


### License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

