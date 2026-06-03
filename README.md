# DplusPy: a tool for inferring multi-population history from ancient DNA

This is the repository for `dpluspy`, a Python package that extends `moments-LD`
(https://github.com/MomentsLD/moments) to study ancient population history
with H<sub>2</sub>.
H<sub>2</sub> is a two-locus genetic statistic, defined as the probability that
two genome copies differ by state at a pair of loci spanning some recombination
fraction.
The package takes its name from an earlier name for the statistic (D<sup>+</sup>).

## Contents

## Installation
You can install `dpluspy` directly from github using `pip`:

```
pip install git+https://github.com/nwcol/dpluspy.git
```

Or you can clone this repository and install the package locally:

```
git clone https://github.com/nwcol/dpluspy.git
cd dpluspy
pip install -r requirements.txt
pip install .
```

If you want to edit the code, it's useful to perform an editable install:

```
pip install -e .
```

## Dependencies



## Examples

### Estimating H<sub>2</sub> from high-coverage data


### Estimating H<sub>2</sub> from low-coverage data


### Inferring demographic parameters


### Quantifying parameter uncertainty


### Performing model choice


## Citing DplusPy
