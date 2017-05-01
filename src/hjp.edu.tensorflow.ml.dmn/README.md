## Dynamic Memory Network Plus in TensorFlow for bAbI data

Dynamic Memory Network Plus (DMN+) implementation in TensorFlow for question answering on the bAbI data.


### Dependencies

Python 2.7.x

TensorFlow 1.0.1

Numpy 1.11.2


### Quickstart

Run the included shell script to fetch the data

	bash data.sh
	
Use 'train.py' to train the DMN+ model contained in 'dmn.py'

	python train.py -b 1
	
After training is finished, test the model on a specified task

	python test.py -b 1
	
	
### Reference

Structure and parameters from:
* [Dynamic Memory Networks for Visual and Textual Question Answering](http://proceedings.mlr.press/v48/xiong16.pdf). Xiong et al., ICML 2016.
* [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](http://proceedings.mlr.press/v48/kumar16.pdf). Kumar et al., ICML 2016.


### Acknowledgments

Our implementation utilizes code from the following:
* [Alex Barron's DMN-Plus repo](https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow)


### License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

