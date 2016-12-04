t-SNE Algorithm
===============

T-SNE stands for t-distributed stochastic neighbor
embedding.

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

T-SNE works in two steps.

__First step: Probability distribution__

For this step, the algorithm constructs a probability
distribution over pairs of high-dimensional objects. With
these distribution, similar pairs have a high probability
of being picked together, where as dissimilar points have
a small change of being picked.

__Second step: Clustering the objects__

In the second step, the algorithm makes a low distribution
map of the objects, and uses the Kullback-Leibler divergence
between the objects on the map. In the original concept of
the algorithm, the Euclidean distance was used.

## Resources

* Wikipedia - [t-distributed stochastic neighbor embedding](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
* Github - Laurens van Maarten - [t-SNE](https://lvdmaaten.github.io/tsne/)
* Paper - [Visualizing Data using t-SNE](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)
* [Scikit-learn t-SNE package](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
* [How to Use t-SNE Effectively](http://distill.pub/2016/misread-tsne/)
* Google AI Experiments - [Visualizing High-Dimensional Space](https://aiexperiments.withgoogle.com/visualizing-high-dimensional-space)
* Google AI Experiments - [Bird Sounds](https://aiexperiments.withgoogle.com/bird-sounds)