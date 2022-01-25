# Introduction

`TrustyAI-Python` is a Python library that leverages the ML/AI explainability features available
on [TrustyAI](https://kogito.kie.org/trustyai/), a part of the [Kogito](https://kogito.kie.org/) project.

## Setup

### PyPi

The simplest way to get started with `TrustyAI-Python` is to install from PyPi with:

```shell
$ pip install trustyai
```

### Local

Alternatively, this library can installed locally by cloning the [main repo](https://github.com/trustyai-python/module) and installing
the minimum build dependencies:

```shell
$ git clone git@github.com:trustyai-python/module.git trustyai-python-module
$ cd trustyai-python-module
$ pip install -r requirements.txt
$ python setup.py build install # -- force if reinstalling
```

If running the examples or developing, also install the development dependencies:

```shell
pip install -r requirements-dev.txt
```

### Docker

Alternatively create a container image and run it using

```shell
$ docker build -f Dockerfile -t $USER/python-trustyai:latest .
$ docker run --rm -it -p 8888:8888 $USER/python-trustyai:latest
```

A Jupyter notebook server will be available at `localhost:8888`.

### Binder

You can also run the example Jupyter notebooks using `mybinder.org`:

- https://mybinder.org/v2/gh/trustyai-python/examples/main

## Getting started

The very first step is to import the module and initialise it.
This is will start the JVM and add the relevant Java libraries to the module's
class path. For instance,

```python
import trustyai

trustyai.init()
```

If the dependencies are not in the default `dep` sub-directory, or
you want to use a custom classpath you can specify it with:
```python
import trustyai

trustyai.init(path="/foo/bar/explainability-core-2.0.0-SNAPSHOT.jar")
```

In order to get all the project's dependencies, the script `deps.sh` can be run and dependencies will
be stored locally under `./dep`.

This needs to be the very first call, before any other call to TrustyAI methods. After this, we can call all other methods, as shown in the examples.

## Writing your model in Python

To code a model in Python you need to write it a function with takes a Python list of `PredictionInput` and
returns a (Python) list of `PredictionOutput`. 

This function will then be passed as an argument to the Python `Model`
which will take care of wrapping it in a Java `CompletableFuture` for you.
For instance,

```python
from trustyai.model import Model

def myModelFunction(inputs):
    # do something with the inputs
    output = [predictionOutput1, predictionOutput2]
    return output

model = Model(myModelFunction)

inputs = [predictionInput1, predictionInput2]

prediction = model.predictAsync(inputs).get()
```

You can see the `sumSkipModel` in the [LIME tests](https://github.com/trustyai-python/module/blob/main/tests/test_limeexplainer.py).

## Examples

You can look at the [tests](https://github.com/trustyai-python/module/tree/main/tests) for working examples.

There are also [Jupyter notebooks available](https://github.com/trustyai-python/examples).