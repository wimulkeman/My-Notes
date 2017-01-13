Machine Learning Dictionary
===========================

# A

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

## Binary neuron

See artificial neuron.

# C

## Chain formula

In calculus the chain formula is used to compute the derivative
of the composition of two or more functions.

- Wikipedia - [Chain rule](https://en.wikipedia.org/wiki/Chain_rule)

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

# D

## Data Augmentation

A process where existing data is slightly modified to create more training
data which represents the same values as the original data.

- R-Bloggers - [What you need to know about data augmentation for machine learning](https://www.r-bloggers.com/what-you-need-to-know-about-data-augmentation-for-machine-learning/)

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

# E

## Explanatory variable

See independent variable.

# F

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

## Gated Recurrent Unit

A GRU is similar to LSTM, but have fewer parameters than a LSTM. That is
because they lack an output gate.

![LSTM / GRU](https://cdn-images-1.medium.com/max/2000/1*K9g9EOeQ9Ca0jdOMmXKrQg.png)

- Wikipedia - (Gated Recurrent Unit)[https://en.wikipedia.org/wiki/Gated_recurrent_unit]

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

## LSTM

See Long Short Term Memory.

# M

## MaxEnt

See Maximum entropy classifier.

## Maximum entropy classifier

See Multinomial logistic regression.

## McCulloch-Pitts Neuron

See artificial neuron.

## MCP Neuron

See McCulloch-Pitts Neuron.

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

## Neural networks

See Artificial Neural Networks.

## NV neuron

See artificial neuron.

# O

## One Shot Learning

See Bayesian Program Learning.

## Output tensor

See Tensor.

# P

## Polytomous LR

See Multinomial logistic regression.

# Q
# R

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

## Softmax regression

See Multinomial logistic regression.

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

# W
# X
# Y
# Z

## Resources

- Medium - [Deep Learning Cheat Sheet](https://hackernoon.com/deep-learning-cheat-sheet-25421411e460#.1fde6uhjf)
- RapidTables - [Mathematical Symbols](http://www.rapidtables.com/math/symbols/index.htm)