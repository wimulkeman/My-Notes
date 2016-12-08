Machine Learning Dictionary
===========================

# A
# B

## Back Propagation

Back Propagation, sometimes shorted to back prop, is the
process of back tracking errors through the weights of the
network after forward propagating inputs through the network.
This is used by applying the chain rule in calculus.

# C

## Chain formula

In calculus the chain formula is used to compute the derivative
of the composition of two or more functions.

Wikipedia - [Chain rule](https://en.wikipedia.org/wiki/Chain_rule)

## CNN

See Convolutional Neural Network.

## ConvNet

See Convolutional Neural Network.

## Convolutional Neural Network

A Convolutional Neural Network s a type of feed-forward artificial
neural network in which the connectivity pattern between its
neurons is inspired by the organization of the animal visual cortex.

Wikipedia - [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)

# D

## Dependent variable

Dependent variables represent the output or outcome whose variation is
being studied.

Wikipedia - [Dependent and independent variables](https://en.wikipedia.org/wiki/Dependent_and_independent_variables)

## Derivative

The derivative of a function of a real variable measure the sensitivity
to change (a function value or dependent variable) which is determined
by another quantity (the independent variable).

Wikipedia - [Derivative](https://en.wikipedia.org/wiki/Derivative)

# E

## Explanatory variable

See independent variable.

# F
# G

## Gradient &nabla;

The gradient, also called Nabla, is the partial derivative of a function that
takes in multiple vectors and outputs a single value
(i.e. our cost functions in Neural Networks). The gradient
tells us which direction to go on the graph to increase
our output if we increase our variable input. We use the
gradient and go in the opposite direction since we want to
decrease our loss.

# H
# I

## Independent variable

Independent variables are the input or causes which are
potential reasons for variation.

Wikipedia - [Dependent and independent variables](https://en.wikipedia.org/wiki/Dependent_and_independent_variables)

# J
# K
# L

## linear predictor function

In statistics and in machine learning, a linear predictor function
is a linear function (linear combination) of a set of coefficients
and explanatory variables (independent variables), whose value is
used to predict the outcome of a dependent variable.

Wikipedia - [Linear predictior function](https://en.wikipedia.org/wiki/Linear_predictor_function)

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

Wikipedia - [Linear regression](https://en.wikipedia.org/wiki/Linear_regression)

## Logistic curve

See Logistic function.

## Logistic function

The Logistic function, or logistic curve is a common "S" shape (sigmoid curve),
with the equation:

![Logistic function](https://wikimedia.org/api/rest_v1/media/math/render/svg/2770ecdecd1a6d2375d17f73013905cea5fb2668)

![Logistic curve](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

Wikipedia - [Logistic function](https://en.wikipedia.org/wiki/Logistic_function)

# M

## Multiple linear regression

See linear regression.

# N

## Nabla

See Gradient.

# O
# P
# Q
# R

## Rectified Linear Units

The sigmoid function has an interval of [0,1], but the ReLU has a range
from [0, infinity]. This makes the sigmoid function better for linear
regression, and the ReLU better for representing positive numbers.

The ReLU does not suffer from the vanishing gradient problem.

## ReLU

See Rectified Linear Units.

# S

## Simple linear regression

See Linear regression.

## Sigma function

See Sigmid function.

## Sigmoid function &sigma; | &sigmaf;

The sigmoid function activates the weights in the neural network
in the interval of [0,1]. The function graphed out is &sigmaf;, but
is sometimes written as &sigma;. The Sigmid function is known as a
logistic function and is best suited for linear regression.

# T
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

Wikipedia - [Vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)

# W
# X
# Y
# Z

## Resources

- Medium - [Deep Learning Cheat Sheet](https://hackernoon.com/deep-learning-cheat-sheet-25421411e460#.1fde6uhjf)