Machine Learning Dictionary
===========================

# A

## Apriori prediction

## Artificial Neural Network

Neural networks are a computational approach which is based on a large
collection of neural units (AKA artificial neurons).

- Wikipedia - [Artificial Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network)

## Artificial neuron

An artificial neuron is a mathematical function conceived as a model of
biological neurons. Artificial neurons are the constitutive units in an
artificial neural network. Depending on the specific model used they may
be called a semi-linear unit, Nv neuron, binary neuron, linear threshold
function, or McCulloch–Pitts (MCP) neuron. The artificial neuron receives
one or more inputs (representing dendrites) and sums them to produce an
output (representing a neuron's axon). Usually the sums of each node are
weighted, and the sum is passed through a non-linear function known as an
activation function or transfer function. The transfer functions usually
have a sigmoid shape, but they may also take the form of other non-linear
functions, piecewise linear functions, or step functions.

- Wikipedia - [Artificial neuron](https://en.wikipedia.org/wiki/Artificial_neuron)

# B

## Back Propagation

Back Propagation, sometimes shorted to back prop, is the
process of back tracking errors through the weights of the
network after forward propagating inputs through the network.
This is used by applying the chain rule in calculus.

## Batch normalization

When a network has many deep layers, they get issues with internal covariate
shift. The shift is “the change in the distribution of network activations
due to the change in network parameters during training.” (Szegedy). Reducing
this shift helps training faster and better. Batch normalization solved this
problem by normalization each batch into the network by both mean and variance. 

## Bayesian Program Learning

A learning mechanism, where the network trains on a low number of
data examples, which could even be one example. The network makes its
guess on what most likely the outcome will be. If the network is incorrect
the weights are recalculated.

- Quora - [What is Bayesian Program Learning (BPL)?](https://www.quora.com/What-is-Bayesian-Program-Learning-BPL)
- Paper - [Human-level concept learning through probabilistic program induction](http://web.mit.edu/cocosci/Papers/Science-2015-Lake-1332-8.pdf)
- Github - [BPL](https://github.com/brendenlake/BPL)
- Github - [PyBPL](https://github.com/MaxwellRebo/PyBPL)
- Paper - [One-shot Learning with Memory-Augmented Neural Networks](https://arxiv.org/pdf/1605.06065v1.pdf)
- Github - [ntm-one-shot](https://github.com/tristandeleu/ntm-one-shot)
- Paper - [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080v1.pdf)

## Bayesian theorem

P(B|A) P(A)<br>
-------------- = P(A|B)<br>
P(B)

or

Number of favorable (desired) outcomes<br>
------------------------------------------------- = Theoretical probabillty<br>
Total number of of possible outcomes

## Bias

Statistical bias is a feature of a statistical technique or of its results
whereby the expected value of the results differs from the true underlying
quantitative parameter being estimated.

![TensorFlow example Bias](https://www.tensorflow.org/images/softmax-regression-scalargraph.png)

b[n] is the bias added to the weight calculation.

## Binary neuron

See artificial neuron.

# C

## Chain formula

In calculus the chain formula is used to compute the derivative
of the composition of two or more functions.

- Wikipedia - [Chain rule](https://en.wikipedia.org/wiki/Chain_rule)

## Cloud AutoML

Platform for Google witch provides pre-trained models for using in
easy-to-use ML setups.

- Google - [Cloud AutoML](https://cloud.google.com/automl/zz)

## CNN

See Convolutional Neural Network.

## Conditional maximum entropy model

See Multinomial logistic regression.

## Connectionist systems

See Artificial Neural Networks.

## ConvNet

See Convolutional Neural Network.

## Convolutional Neural Network

A Convolutional Neural Network s a type of feed-forward artificial
neural network in which the connectivity pattern between its
neurons is inspired by the organization of the animal visual cortex.

See also Artificial Neural Network.

- Wikipedia - [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)

## Cost

Used as a term to describe how far off the network is from the desired outcome. Also called loss.
The lower the cost score, the better.

The cross-entropy function is often used to calculate the cost or loss.

## Cross-entropy

Cross entropy is used to calculate how far off the label prediction is.
It is a loss function which is related to the entropy of thermodynamics
concept of entropy. It is used in the multiclass classification to find
an error in the prediction.

![Cross entropy function](https://cdn-images-1.medium.com/max/1600/1*9ZBskBY_piVwqC4GdZRl8g.png)

# D

## Data Augmentation

A process where existing data is slightly modified to create more training
data which represents the same values as the original data.

- R-Bloggers - [What you need to know about data augmentation for machine learning](https://www.r-bloggers.com/what-you-need-to-know-about-data-augmentation-for-machine-learning/)

## Deep learning

Buzzword used for Artificial Neural Networks. See Artificial Neural Networks.

## Dependent variable

Dependent variables represent the output or outcome whose variation is
being studied.

- Wikipedia - [Dependent and independent variables](https://en.wikipedia.org/wiki/Dependent_and_independent_variables)

## Derivative

The derivative of a function of a real variable measure the sensitivity
to change (a function value or dependent variable) which is determined
by another quantity (the independent variable).

- Wikipedia - [Derivative](https://en.wikipedia.org/wiki/Derivative)

## Directed cycle

A directed cycle is a cycle which can only be followed in one direction.
Except for the start and end point, none of the vertices and edges may be
traversed more than once until the walk is completed.

## Drop-out

Drop is a method used to prevent overfitting within a network, and gives
the opportunity to combine multiple types of network. Within this method
visible and hidden units are picked randomly from the network and dropped.

Mostly this is done by giving a percentage for a layer for the drop-out.

![Drop-out in a Neural Network](https://cdn-images-1.medium.com/max/1600/1*XkDC2Iwb9jSyRIWBUoDFtQ.png)

# E

## Explanatory variable

See independent variable.

# F

## F1/F Score

The measurement of how accurate a model is by using the calculation
of the precision and recall within the following formula.

F1 = 2 * (Precision * Recall) / (Precision + Recall)

## Factorial (2!)

The factorial of a number is all the numbers multiplied up to that number.
The factorial of 3 = 3! = 1 x 2 x 3.

## Feedforward neural network

A feedforward neural network is a artificial neural network where the
connections between the units do not form a cycle.

This type of nueral network was the first and simplest type of artificial
neural networks.

- Wikipedia [Feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network)

# G

## GAN

See General Adversarial Network.

## Gated Recurrent Unit

A GRU is similar to LSTM, but have fewer parameters than a LSTM. That is
because they lack an output gate.

![LSTM / GRU](https://cdn-images-1.medium.com/max/2000/1*K9g9EOeQ9Ca0jdOMmXKrQg.png)

- Wikipedia - [Gated Recurrent Unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit)

## Generative Adversarial Network

The approach, known as a generative adversarial network, or GAN,
takes two neural networks—the simplified mathematical models of the
human brain that underpin most modern machine learning—and pits them
against each other in a digital cat-and-mouse game.

Both networks are trained on the same data set. One, known as the
generator, is tasked with creating variations on images it’s already
seen—perhaps a picture of a pedestrian with an extra arm. The second,
known as the discriminator, is asked to identify whether the example
it sees is like the images it has been trained on or a fake produced
by the generator—basically, is that three-armed person likely to be real?

Over time, the generator can become so good at producing images that
the discriminator can’t spot fakes. Essentially, the generator has
been taught to recognize, and then create, realistic-looking images
of pedestrians.

The technology has become one of the most promising advances in AI in
the past decade, able to help machines produce results that fool even humans.

- ARXIV - [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf)
- GitHub - [The GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)
- Technology Review - [10 BREAKTHROUGH TECHNOLOGIES 2018 > Dueling Neural Networks](https://www.technologyreview.com/lists/technologies/2018/)

## GloVe

Equilevent of word2vec, but works a bit faster.

## GNMT

See Google Neural Machine Translation system.

## Google Neural Machine Translation system

The translation system build by Google which uses Machine Learning for
its translation. One of the methods used is Recurrent Neural Networks.

- Google Research Blog - [A Neural Network for Machine Translation, at Production Scale](https://research.googleblog.com/2016/09/a-neural-network-for-machine.html)

## Gradient &nabla;

The gradient, also called Nabla, is the partial derivative of a function that
takes in multiple vectors and outputs a single value
(i.e. our cost functions in Neural Networks). The gradient
tells us which direction to go on the graph to increase
our output if we increase our variable input. We use the
gradient and go in the opposite direction since we want to
decrease our loss.

## GRU

See Gated Recurrent Unit.

# H

## Haar cascade

The name for a method which is used for object detection.

- Python Programming - [Cascade OpenCV Python Tutorial](https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/)
- Wikipedia - [Haar-like features](https://en.wikipedia.org/wiki/Haar-like_features)

# I

## Image Augmentation

See data augmentation.

## Independent variable

Independent variables are the input or causes which are
potential reasons for variation.

- Wikipedia - [Dependent and independent variables](https://en.wikipedia.org/wiki/Dependent_and_independent_variables)

## Inductive transfer

See transfer learning

# J
# K

## K-Nearest neighbor

## Keras

Keras is a high level neural network which uses eather Tensorflow, or Theano.
It became the default higher level layer for Tensorflow since v1.0.

- [Keras](https://keras.io/)

# L

## L1 and L2 regularization

These regularization methods prevent overfitting by imposing a penalty
on the coefficients. L1 can yield sparse models while L2 cannot.
Regularization is used to specify model complexity. Because of this,
models generalize better and don't become overfitted with the training
data.

## linear predictor function

In statistics and in machine learning, a linear predictor function
is a linear function (linear combination) of a set of coefficients
and explanatory variables (independent variables), whose value is
used to predict the outcome of a dependent variable.

- Wikipedia - [Linear predictior function](https://en.wikipedia.org/wiki/Linear_predictor_function)

## Linear regression

In statistics, linear regression is an approach for modeling the
relationship between a scalar dependent variable y and one or more
explanatory variables (or independent variables) denoted X. The case
of one explanatory variable (independent variable) is called simple
linear regression. For more than one explanatory variable (independent
variable), the process is called multiple linear regression.

In linear regression, the relationships are modeled using linear
predictor functions whose unknown model parameters are estimated from
the data. Such models are called linear models.

- Wikipedia - [Linear regression](https://en.wikipedia.org/wiki/Linear_regression)

## Linear threshold function

See artificial neuron.

## Logistic curve

See Logistic function.

## Logistic function

The Logistic function, or logistic curve is a common "S" shape (sigmoid curve),
with the equation:

![Logistic function](https://wikimedia.org/api/rest_v1/media/math/render/svg/2770ecdecd1a6d2375d17f73013905cea5fb2668)

![Logistic curve](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

- Wikipedia - [Logistic function](https://en.wikipedia.org/wiki/Logistic_function)

## Long Short Term Memory

LSTM is often used in Recurrent Neural Networks, but can be used in other
implementations. They are used as little memory units that keep state between
the inputs for training the network. They also help solving the vanishing
gradient problem.

![LSTM / GRU](https://cdn-images-1.medium.com/max/2000/1*K9g9EOeQ9Ca0jdOMmXKrQg.png)

- Wikipedia - [Long Short Term Memory](https://en.wikipedia.org/wiki/Long_short-term_memory)

## Loss

See cost.

## Loss function

See Objective function.

## LSTM

See Long Short Term Memory.

# M

## Matrix

An matrix is a 2D array of numbers. Instead of like a vector, in a
matrix a single element is identified by two indexes.

## MaxEnt

See Maximum entropy classifier.

## Maximum entropy classifier

See Multinomial logistic regression.

## McCulloch-Pitts Neuron

See artificial neuron.

## MCP Neuron

See McCulloch-Pitts Neuron.

## MANN

See Memory Augmented Neural Network.

## Memory Augmented Neural Network

The implementation of One-Shot learning, but with the usage of a neural network.
Introduced with a paper by Google DeepMind.

A MANN consists of two parts. The first part is the controller and is a feed-forward
network, or a LSTM network. The second part is a external memory module.	

- arXiv - [One-shot Learning with Memory-Augmented Neural Networks](https://arxiv.org/abs/1605.06065)

## Multiclass LR

See Multinomial logistic regression.

## Multinomial logistic regression

In statistics, multinomial logistic regression is a
classification method that generalizes logistic regression
to multiclass problems, i.e. with more than two possible
discrete outcomes.

- Wikipedia - [Multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)

## Multinomial logit

See Multinomial logistic regression.

## Multiple linear regression

See linear regression.

# N

## Nabla

See Gradient.

## Natural Language Processing

A field of study in which natural spoken or written language is used to
retrieve the users intended actino.

## Neural networks

See Artificial Neural Networks.

## NLP

See Natural Language Processing.

## NV neuron

See artificial neuron.

# O

## Objective function

The goal of the function in a network is to minimize the loss so to
maximize the accuracy of the network.

## One-hot vector

A one-hot vector is a vector which is 0 in most dimensions, and 1 in
a single dimension. For example [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].

## One Shot Learning

See Bayesian Program Learning.

## Optimization function

See Objective function.

## Output tensor

See Tensor.

# P

## Polytomous LR
    
See Multinomial logistic regression.


## Precision

Precise is the calculation of every prediction, which were actually
positive.

Precision = true positives / (true positives + false positives).

# Q
# R

## Recall

Recall is the calculation of all that have positive predictions, which
ones actually were positive.

Recall = True Positives / (True Positives + False Negatives).

## Rectified Linear Units

The sigmoid function has an interval of [0,1], but the ReLU has a range
from [0, infinity]. This makes the sigmoid function better for linear
regression, and the ReLU better for representing positive numbers.

The ReLU does not suffer from the vanishing gradient problem.

## Recurrent Neural Network

A recurrent neural network is a class of artificial neural network where
connections between units form a directed cycle. This creates an internal
state of the network which allows it to exhibit dynamic temporal behavior.
Unlike feedforward neural networks, RNNs can use their internal memory to
process arbitrary sequences of inputs. This makes them applicable to tasks
such as unsegmented connected handwriting recognition or speech recognition.

- Wikipedia - [Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network)

## ReLU

See Rectified Linear Units.

## RNN

See Recurrent Neural Network.

# S

## Scalar

A single number.

## Semi-linear unit

See artificial neuron.

## Simple linear regression

See Linear regression.

## Sigma function

See Sigmid function.

## Sigmoid function &sigma; | &sigmaf;

The sigmoid function activates the weights in the neural network
in the interval of [0,1]. The function graphed out is &sigmaf;, but
is sometimes written as &sigma;. The Sigmid function is known as a
logistic function and is best suited for linear regression.

## Softmax

Softmax is a function usually called at the end of a Neural Network
for classification. This functions does a multinomial logistic regression
and is generally used for multi class classification. Cross entropy is
often used in combination as a loss function.

Softmax is often used as the final layer in a network to assign probabilities
to a object if the object can be different things. All the given probabilities
should add up to 1.

y = softmax(evidence)

The equation would look like the following:

![TensorFlow example Softmax equation](https://www.tensorflow.org/images/softmax-regression-scalarequation.png)

Or vectorized:

![TensorFlow example Softmax equation](https://www.tensorflow.org/images/softmax-regression-vectorequation.png)

Which can be written as y = softmax(Wx + b).

- Neural and network deep learning [CHAPTER 3 - Improving the way neural networks learn](http://neuralnetworksanddeeplearning.com/chap3.html#softmax)

## Softmax regression

See Multinomial logistic regression.

## Stochastic gradient decent

A form of Stochastic training which is combinend with gradient decent to derive
the right value for the wieghts and bias.

## Stochastic training

Using small batches of random data to train the neural network.

# T

## Tanh

Tanh is a function used to initialize the weights of a network as [-1,1].
The better the data is normalized, the stronger the gradient wil be. When
the data becomes more centered around the 0, the derivative will be higher.

![Tanh function](https://cdn-images-1.medium.com/max/2000/1*QYeGYddNRbrBJjkNxzw9FQ.png)

## Tensor

In Machine Learning, tensors are the multidimensional arrays of data
that flows through the learning network.

Based on the data, a output tensor is created. This is the result of
the processing of the tensors.

- Stack Exchange - [Why the sudden fascination with tensors?](http://stats.stackexchange.com/a/198395)
- Youtube - [Tensorboard Explained in 5 Min](https://www.youtube.com/watch?v=3bownM3L5zM)

## Tensorboard

Tensorboard is a package bundles with Tensorflow, which visualizes the learning
process of a learning network. This is ideal for debugging and optimizing because
it also enables to see stats like memory and cpu usage en processing time.

- Youtube - [Tensorboard Explained in 5 Min](https://www.youtube.com/watch?v=3bownM3L5zM)
- Tensorflow - [Tensorboard: Visualize learning](https://www.tensorflow.org/versions/r0.12/how_tos/summaries_and_tensorboard/index.html)

## Tensorflow

A Machine Learning framework open sourced by Google.

- [Tensorflow](https://www.tensorflow.org)

## Training loss

The prediction of how well the RNN predicted the next part, while being fed
training data. Lower score is better.

## Transfer learning

Using the knowledge of a previous solved problem, and use this knowledge
(for example image recognition) for a different but related problem.

- Wikipedia - [Inductive transfer](https://en.wikipedia.org/wiki/Inductive_transfer)

## T-distributed stochastic neighbor

The t-SNE algorithm is a machine learning algorithm
used to visualize high-dimensional data into a two or
three dimension space. The algorithm uses dimensionality
reduction for this, and is a nonlinear dimensionality
reduction technique.

It is used to find similarities in text, sound and other
objects in a variety fields:
* Computer science
* Cancer research
* Music/sound analysis (for example bird sounds)

Further description is found under My Notes > t-SNE Algorithm
- My Notes - [t-SNE Algorithm](t-SNE%20Algorithm)

## T-SNE

See t-distributed stochastic neighbor.

# U
# V

## Validation loss

In some training scenarios for a RNN, the network generates a set of data on its
own which should resemble the validation sequence it is changed on. This is than
mapped to the real validation data set. The lower the score the better.

## Vanishing gradient problem

In machine learning, the vanishing gradient problem is a difficulty
found in training artificial neural networks with gradient-based
learning methods and back propagation. In such methods, each of the
neural network's weights receives an update proportional to the gradient
of the error function with respect to the current weight in each iteration
of training. Traditional activation functions such as the hyperbolic
tangent function have gradients in the range (−1, 1) or [0, 1), and
back propagation computes gradients by the chain rule. This has the effect
of multiplying n of these small numbers to compute gradients of the "front"
layers in an n-layer network, meaning that the gradient (error signal)
decreases exponentially with n and the front layers train very slowly.

By Recurrent Neural Networks and a vanishing gradient problem, the
networks begins to lose the context of the input prior after about 7 steps.

- Wikipedia - [Vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)

## Vector

An array of numbers where each element is identified by a single index.

# W

## Weights

Weights are used within neural networks to derive the possibility of what a
objects class could be.

When the weight is positive, it afirms the change of a object being of a certain
class. When the weight is negative, it is evidence against being from that class.

## Word2vec

Word2vec is a technique introduced by Google to replace words with vectors. These
vectors can then be used to train a neural network within a NLP field.

# X
# Y
# Z

## Resources

- Medium - [Deep Learning Cheat Sheet](https://hackernoon.com/deep-learning-cheat-sheet-25421411e460#.1fde6uhjf)
- RapidTables - [Mathematical Symbols](http://www.rapidtables.com/math/symbols/index.htm)

# Other glossaries - dictionaries

- [Google Machine Learning Glossary](https://developers.google.com/machine-learning/glossary/)