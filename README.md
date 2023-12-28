# Yoga-82 Visual Tranformer Classifier - by Pipe

## Setting Up

### Setup a Python3 Enviroment

It's recommended to work on a Python 3 enviroment so that dependencies and libraries don't have version missmatching with other projects. You can use your prefered way to create your own enviroment. In case you don't know how, here is a way to do it with Anaconda.

Anaconda is an open source distribution for the Python programming language. The Anaconda distribution includes many of the most commonly used Python libraries by default, also a user interface for managing and updating packages.

You can create your own working environment so that, depending on the project, you can use different dependencies packages.

1. Install [anaconda](https://www.anaconda.com/) and [git](https://git-scm.com)
1. Open anaconda terminal
1. Type ``conda create -n vit python=3.9`` create working environment
1. Type ``conda activate vit``


If executing on windows, you might as well:

    conda install -c menpo wget

### Get the Project and requirements

1 ) Clone the repository and cd into it

```
git clone https://github.com/pipperv/yoga82-vit.git
cd yoga82-vit
```

2 ) Install the prerequisites

```
pip install -r requirements.txt
```

3 ) Install Pytorch

With Conda:

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Or pip

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


### Download Dataset

Excecute the ``download_dataset.py`` file with python.

```
python download_dataset.py
```

There is a BlackList .txt file inside Yoga-82/ that lists files didn't correctly downloaded. This is so you don't have to try download corrupted files. In case you want to try download all dataset files or use a different Yoga-82 version, just delete ``Yoga-82/black_list.txt`` content.

### Current Progress

``vit.py`` contains all necesary modules to create a Vision Transformer Clasificator, everything built around Pytorch nn Modules for easy integration and training.
``dataset.py`` and ``dataset_utils.py`` have all necesary tools to create and use the Yoga-82 dataset. ``test.ipynb`` contains the code done for training the ViT, including Dataset creation and preparation, with data aumentation, normalization and dataset balance.

### ImageNet Pretrained Model

To use the weights of the model pretrained in imagenet, download the model from this [link](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) and put the file inside the main folder. Make shure to set ``"load_pretrained": True`` in the config dictionary of the test notebook.

