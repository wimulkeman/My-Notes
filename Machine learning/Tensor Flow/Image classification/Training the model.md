Training the model
==================

The expected structure for the setup where the command is used, is

```
./tensor_flow_image_classification
    run_tf_image_classification.sh
    tf_files
        images
            image_label_1
                image1.jpg
                image2.jpg
                ....
            image_label_2
                image3.jpg
                image4.jpg
                ....
        label_image.py  # The python file for labeling images after training the model
```

The content of the sh file is used to run a docker image for the training
of the TF image classification.

The content of run_tf_image_classification.sh

```bash
docker run -it -d \
    -v $(pwd)/tf_files:/tf_files \
    --name tensorflow \
    gcr.io/tensorflow/tensorflow:latest-devel
```

For the use within most of the tutorials, the use of the **latest-devel** image is
important. In this image the tensorflow directory is available in the root of the
docker image. In the other versions of the immage, tensorflow is available as a
binary file in the root.

# 1. Running the command

The command to train the Tensor Flow image classifier is

```bash
python /tensorflow/tensorflow/examples/image_retraining/retrain.py \
    --bottleneck_dir=/tf_files/bottlenecks \
    --how_many_training_steps=500 \
    --model_dir=/tf_files/inception \
    --output_graph=/tf_files/retrained_graph.pb \
    --output_labels=/tf_files/retrained_labels.txt \
    --image_dir=/tf_files/images
```

## The options used

### bottleneck_dir

TF uses this dir to store the cached files of the generated
lower layers so those don't need to be recalculated each run.

### how_many_training_steps

The number of cycles TF goes through the images to train itself.

### model_dir

This one is very important. It point TF to the directory where to
store the model generated for the image recognition. This model is
the part used to recognize images later on.

### output_graph

This generates a graph which can be displayed in Tensor Board. The
graph will make the layers visible that are created.

### output_labels

These will be the name of labels which are the same as the directory names
in the images directory, and which are the labels who are connected to
the images trained with.

### image_dir

The directory where TF has to look for the images which needs to be labeled.
The name of the subdirectories within this directory are the names of the
labels for the images within them.

# 2. Determine the precision

After the command has been run, there is a number shown which is the expected
precision of the trained model.

```bash
2016-10-22 02:29:13.553649: Step 470: Train accuracy = 90.0%
2016-10-22 02:29:13.553737: Step 470: Cross entropy = 0.334267
2016-10-22 02:29:13.898204: Step 470: Validation accuracy = 81.0%
2016-10-22 02:29:17.366515: Step 480: Train accuracy = 95.0%
2016-10-22 02:29:17.366603: Step 480: Cross entropy = 0.282858
2016-10-22 02:29:17.712904: Step 480: Validation accuracy = 83.0%
2016-10-22 02:29:21.181999: Step 490: Train accuracy = 97.0%
2016-10-22 02:29:21.182090: Step 490: Cross entropy = 0.257443
2016-10-22 02:29:21.575653: Step 490: Validation accuracy = 80.0%
2016-10-22 02:29:24.680511: Step 499: Train accuracy = 93.0%
2016-10-22 02:29:24.680597: Step 499: Cross entropy = 0.356517
2016-10-22 02:29:25.039826: Step 499: Validation accuracy = 75.0%
Final test accuracy = 92.6%
Converted 2 variables to const ops.
```

This number is the expected accuracy for all the labels combined. The
precision per label can be higher or lower.

# 3. Write a script to classify images

To classify a image, the im age is directed to a python file. This contains
the following code.

```python
import tensorflow as tf, sys

image_path = sys.argv[1]

# The path where to find the image
# The file is provided by python image_classifier.py ./images/image2.jpg
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Load the generated label files for labeling the image
# the rstrip prevent that return characters are also read
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Load the im age and predict the label
with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor,
             {'DecodeJpeg/contents:0': image_data})

    # Use the outcome of the prediction to show the first label that could
    # be it, followed by the next option, etc
    # The labels are ordered by confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    # Take the list with scores, and print them out
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
```

# 4. Run the classification on a image

Run the following command to classify a image

```bash
python /tf_files/label_image.py /tf_files/images/image_label1/image1.jpg
```

This will output a list met prediction scores which will give the scores for
the labels which are defined.

# Troubleshooting

### Cannot import name graph_util

When running a new version of the TF repo, you could encounter the following
error when running the command.

```python
Traceback (most recent call last):
  File "tensorflow/examples/image_retraining/retrain.py", line 82, in <module>
    from tensorflow.python.framework import graph_util
ImportError: cannot import name graph_util
```

This is a result of a recent move made in the TF library from the graph_util
file from the client directory to the framework directory.

To temporary fix this, change the following line in the file retrain.py
(/tensorflow/tensorflow/examples/image_retraining/retrain.py)

```python
from tensorflow.python.framework import graph_util
```

into

```python
from tensorflow.python.client import graph_util
```