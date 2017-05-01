# DMN in TF
Dynamic Memory Network implementation in TensorFlow for question answering on the bAbI 10k dataset.


## Usage
Install [TensorFlow r1.0](https://www.tensorflow.org/install/)

Use 'train.py' to train the DMN model contained in 'dmn.py'

	python train.py --b 1

Once training is finished, test the model on a specified task

	python test.py --b 2
