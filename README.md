
# 2018 UIdaho Student Data Science Competition

The [machine learning competition](https://dscomp.ibest.uidaho.edu/) will ask participants to grade 20,000 handwritten math quizzes. Can you develop a model that performs better than a random guess? Will you climb to the top of our Leaderboard?

Three prizes will be awarded to the top scorers:

  * Golden Vandal: a University of Idaho scholarship of $500.
  * Silver Vandal: a University of Idaho scholarship of $300.
  * Iron Vandal: a University of Idaho scholarship of $200.

## This repo

Here is the code for one of my more successful models for the student data science competition. It is built on PyTorch, and uses the [PyTorchNet](https://github.com/human-analysis/pytorchnet) basic framework to load a malleable configuration and training/testing structure. That makes it a bit harder to dive into than a single long file unfortunately. 

This model is based on VGG net. 

### Methodology

The idea is pretty simple, use a large VGG like network with some minor randomly applied transformations to the train images.

* Train single network on full set of numbers
* Segment testing images into even fifths (i.e. `[1] [+] [2] [=] [4]`)
* Predict numbers (generate a probability vector)
* For positions 2,4 only take max probability of symbols (i.e. >= 10)
* For other positions use only numbers (i.e. < 10)
* Use predicted numbers/symbols to check if equation is valid
* Set answers to 1 if true, else 0

### Files

The most important files are below:

* models/vgg.net - contains the model architecture used
* datasets/uisdsc.py - Automatically downloads the dataset and performs transformations for the model
* main.py - simply grabs the dataloader and runs the testing and training in a loop.
* train.py - the bulk training function.
* test.py - how the test data gets evaluated

## Where is the full model?

For my final solution I trained three models: 
 * this VGG model
 * A ResNet-like model
 * A densenet-like model
 
However, they were each trained individually using a similar framework and in the end I wrote a sloppy script that took the output files I had copied/pasted into another directory to merge their weighted results. The whole framework was pretty poorly done and obviously more than a little bit of a hackjob. If anyone really wants it, I can take some time to clean it up and post it.

## Usage

1. Spin up visdom `python -m visdom.server`
2. Edit/view `args.txt` to change any preferences
3. Run it: `python main.py`


## Other solutions
[Camden Clark](https://github.com/CamdenClark/FirstDataScienceCompetition) - The winner behind us all.
