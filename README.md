# Introduction to the course
This is the website for the 
[Inria Academy course](https://www.inria-academy.fr/formation/skrub-like-a-pro-clean-prepare-and-transform-your-data-faster/)
on the [skrub package](https://skrub-data.org/stable/): it contains all the material
used for the course, including the datasets and exercises used during the session. 

## Structure of the course
The course covers the main features of skrub, from data exploration to pipeline 
construction. While skrub DataOps are a major feature of the package, they are 
also expansive enough to deserve their own course, and as such only a short introduction
will be given here. 

Each chapter includes a section that describes how a specific feature may assist
in building a machine learning pipeline, along with practical code examples, and
a quiz at the end. 

The course is split in sections, which group relevant material together. Each 
section is wrapped up by an exercise that covers what has been explained in the 
section.
These exercises are made available in `content/exercises` as `py` files, and 
in `content/notebooks` as Jupyter notebooks. 

# Prepration and setup

## Using Jupyterlite
The easiest way to work on the exercises is simply by using Jupyterlite: this 
will create a notebook interface directly from the browser that allows to run the
exercises without needing to create a local environment. 

## Setting up a local environment
If you still want to work locally (for example, if you want to use your own IDE),
you can still do so by cloning the [GitHub repo](https://github.com/skrub-data/skrub-tutorials/tree/main) 
of this book to have access to the exercises. 

### Finding the material
Following any of the following commands should let you open a Jupyter lab or 
notebook instance in the root of the folder. Then, you will find all the course 
material as notebooks in `content/notebooks`, and only the exercises in 
`content/exercises`. 

All the datasets are made available to the notebooks by cloning the repo. 

### Using pixi
The easiest way to set up the environment is by installing and
using [pixi](https://pixi.sh/latest/installation/). Follow the platform-specific 
instructions in the link to install pixi, then open a terminal window. 

Run 
```sh
pixi install
```
to create the environment, followed by 

```sh
pixi run lab
``` 
to start a Jupyter lab instance. 

### Using `pip`
Create the and activate the environment:

```sh
python -m venv skrub-tutorial
source skrub-tutorial/bin/activate
```

Install the required dependencies using the `requirements.txt` file:
```sh
pip install -r requirements.txt
```

Start the Jupyter lab instance: 
```sh
jupyter lab
```

### Using conda
An `environment.yaml` file is provided to create a conda environment. 

Create and activate the environment with

```sh
conda env create -f environment.yaml
conda activate skrub-tutorial
```

Then, start a jupyter lab instance:

```sh
jupyter lab
```

### Using `uv`
Create the environment using `pyproject.toml` as the requirement file. 

```sh
uv venv 
uv pip install -r pyproject.toml
```

Activate the environment that was created in the folder. 
```sh
source .venv/bin/activate
```

Start the Jupyter lab instance: 
```sh
jupyter lab 
```