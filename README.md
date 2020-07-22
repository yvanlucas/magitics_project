# MAGITICS
> Research project using machine learning for prediction of antibiotic resistance of bacterias.

## Table of Contents 

- [Installation](#installation)
- [Usage](#usage)
- [Input format](#input-format)
- [Output](#output)
- [Install Gerbil](#install-gerbil)
- [Contact](#contact)
## Installation

### Requirements
- `Gerbil` API is required to get started (see [Install Gerbil](#Install Gerbil))
- Python3 packages needed
```shell
$ Pandas
$ Scikit-learn
$ Numpy
$ python-Levenshtein
```

### Clone

```shell
$ git clone https://github.com/yvanlucas/magitics
```
## Usage

## Input format

## Output

## Install Gerbil
Gerbil can be found <a href="https://github.com/uni-halle/gerbil.git" target="_blank">here</a>.

The version of Gerbil used (1.1) is developed for Linux operating systems.

1. Install 3rd-party libraries and necessary software:

        sudo apt-get install git cmake g++ libboost-all-dev libz3-dev libbz2-dev

2. Download the Source Files. 

        git clone https://github.com/uni-halle/gerbil.git
        
3. Compile the Sources. Gerbil comes with a CMake Script that should work for various operating systems. CMake will automatically detect whether all mandatory and optional libraries are available at your system.

        cd gerbil
        mkdir build
        cd build
        cmake ..
        make

The `build` directory should now contain a binary `gerbil`.

## Contact

[Dr. Yvan Lucas](mailto:yvanlucas44@gmail.com)
