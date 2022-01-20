[Chinese](docs/README_CN.md)

<p align="center">
  <img src="https://raw.githubusercontent.com/iGame-Lab/TS-VIS/master/docs/images/logo.svg" alt="logo"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-361"><img src="https://img.shields.io/badge/Python-%3E=3.6-blue.svg" alt="python"/></a>
  <a href="https://github.com/iGame-Lab/TS-VIS/blob/master/LICENSE"><img src="https://img.shields.io/github/license/iGame-Lab/TS-VIS" alt="license"/></a>
  <a href="https://pypi.org/project/tsvis"><img src="https://img.shields.io/pypi/v/tsvis" alt="pypi"/></a>
  <a><img src="https://img.shields.io/github/workflow/status/iGame-Lab/TS-VIS/Build%20tsvis" alt="build"></a>
</p>

TS-VIS(Tianshu Visualization) is a visualization tool kit of <a href="https://gitee.com/zhijiangtianshu/Dubhe" target="_blank"> Tianshu AI Platform. </a>, 
which support visualization of the most popular deep learning frameworks, such as TensorFlow, PyTorch, OneFlow, etc.

**[Document (Chinese): https://feyily.github.io/tsvis-document/](https://feyily.github.io/tsvis-document/)**

![](https://raw.githubusercontent.com/iGame-Lab/TS-VIS/master/docs/images/demo.gif)

## HighLights

* Framework-independent, support visualization of the most popular deep learning frameworks, such as TensorFlow, PyTorch, OneFlow, etc.
* Faster response speed
* Support the visualization of large-scale data
* Support real-time visualization during training
* Support embedding sample visualization
* Support neural network exception visualization

## Features

- Graph: Visualize neural network structure, including computational graph and structure graph
- Scalar: Visualize arbitrary scalar data including `accuary` and `loss`
- Media: Visualize media data including images, text, and audio
- Distribution: Visualize the distribution of weights, biases, etc. in neural network
- Embedding: Visualize arbitrary high-dimensional data through dimensionality reduction algorithm
- Hyperparameter: Visualize neural network indicators under different hyperparameters
- Exception: Map neural network tensor data to two dimensions, visualize tensor data statistics
- Custom: Move the charts in `Scalar`, `Media`, and `Distribution` to this module for comparison and viewing

## Install

We provide two installation methods: install by pip and install from source. 
No matter which method you pick, you need to make sure that your Python version is 3.6 or higher, 
otherwise please upgrade Python first.

### Install by pip

```
pip install tsvis
```

### Install from source

TS-VIS adopts the architecture of separation of frontend and backend, 
so you need to build the frontend and backend separately

- **Build frontend from source:**

  ```
  cd webapp
  ```

  Install dependencies first

  ```
  npm install
  ```

  Package frontend to generate static files

  ```
  npm run build
  ```

- **Build backend from source:**

  To install the backend, you need to first move the static files generated by previous step to `tsvis/server/frontend` folder

  Then install the Python dependency package `setuptools`
  
  ```
  pip install setuptools
  ```
  
  Run `setup.py` to install TS-VIS to your Python environment
  
  ```
  python setup.py install
  ```

### Run

After installation, you can run the following command. If the version information is output in the console, it means that you have installed TS-VIS correctly.

```
tsvis -v
```

Then you can run the visualization with the following command

```
tsvis --logdir path/to/logdir/
```

By default, the visualization service will start at `http://127.0.0.1:9898`, open the browser to access the visualization content.

