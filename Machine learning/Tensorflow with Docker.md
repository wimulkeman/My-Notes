Tensorflow with Docker
======================

# Which docker images to use

For development use, the [version]-devel or [version]-devel-gpu images
form tensorflow/tensorflow. These contain a git instance of TensorFlow
and provides the example directories.

# Combining with nvidia-docker

For running with the additional use of a NVidia graphics card, the command
nividia-docker run can be used, which has the same parameters as the
docker run command.

# Errors when docker is upgraded

After upgrading starting the nvidia-docker command can give a error on a
onsupported Docker version. Can be fixed by removing nividia-docker

```bash
sudo apt purge nvidia-docker
```

Make sure there are no containers connected to images tied to nvidia docker.

# Running Jupyter Notebook

Log into the container and run the command:

```bash
jupyter notebook
```

After this the noteboak interface is available on port 8888 (when exposed).

# Running Tensorboard

Log into the container and run the command

```bash
tensorboard path/to/log/directory
```

