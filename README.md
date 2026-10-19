# Introduction to the course

This is the website for the
[Inria Academy course](https://www.inria-academy.fr/formation/skrub-like-a-pro-clean-prepare-and-transform-your-data-faster/)
on the [skrub package](https://skrub-data.org/stable/): it contains all the material
used for the course, including the datasets and exercises used during the session.

The material is available in two formats:

- a **book**, in `book/`, with more detailed explanations;
- a set of **slides**, in `slides/`, for instructor-led sessions.

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
These exercises are stored in `content/exercises` as `py` files; running
`pixi run materials` converts them to Jupyter notebooks in `content/notebooks`
and puts the datasets in `content/data`, ready to be used.

# Preparation and setup

## Using Jupyterlite
The easiest way to work on the exercises is simply by using Jupyterlite: this
will create a notebook interface directly from the browser that allows to run the
exercises without needing to create a local environment.

## Setting up a local environment
If you still want to work locally (for example, if you want to use your own IDE),
you can clone the [GitHub repo](https://github.com/skrub-data/skrub-tutorials/tree/main)
to have access to the exercises.

### Finding the material
All the datasets are made available to the notebooks by cloning the repo; the
exercises are stored in `content/exercises`.

Run
```sh
pixi run materials
```
to generate the notebooks in `content/notebooks` (using
[jupytext](https://jupytext.readthedocs.io/en/latest/)) and copy the datasets to
`content/data`.

### Using pixi
The easiest way to set up the environment is by installing and
using [pixi](https://pixi.sh/latest/installation/). Follow the platform-specific
instructions in the link to install pixi, then open a terminal window in the
root of the repository and run

```sh
pixi install
pixi run materials
pixi run lab
```
to create the environment, generate the notebooks and datasets, and start a
Jupyter lab instance.

### Using `pip`
Create and activate the environment:

```sh
python -m venv skrub-tutorial
source skrub-tutorial/bin/activate
```

Install the required dependencies using the `requirements.txt` file:
```sh
pip install -r requirements.txt
```

Start the Jupyter lab instance from the root of the repository:
```sh
jupyter lab
```

# Building the website locally
The website (landing page, book, slides and the JupyterLite instance) is built
with [quarto](https://quarto.org/) and published on GitHub Pages by a GitHub
Actions workflow on every push to `main`.

To rebuild the full website locally, run

```sh
pixi run render
pixi run serve
```
then open `http://localhost:8000` in a browser. The rendered site is placed in
the `_build` directory.