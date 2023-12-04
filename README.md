## Steps

### Setup with Anaconda🐍

Anaconda is an open source distribution for the Python programming language. The Anaconda distribution includes many of the most commonly used Python libraries by default, also a user interface for managing and updating packages.

You can create your own working environment so that, depending on the project, you can use different dependencies packages.

1. install [anaconda](https://www.anaconda.com/) and [git](https://git-scm.com)
1. open anaconda terminal
1. ⌨️=``conda create -n vit python=3.9`` create working environment
1. ⌨️=``conda activate vit``


If executing on windows, you might as well:

    conda install -c menpo wget

### Usage

1 ) Clone the repository and cd into it

```
git clone https://github.com/pipperv/yoga82-vit.git
cd yoga82-vit
```

2 ) Install the prerequisites

```
pip install -r requirements.txt
```

### Download Dataset