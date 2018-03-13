"""
Script to practice lesson 1
"""
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# define globals
PATH = "data/dogscats"
size = 224

arch = resnet34
# convert dataset into a fastai native dataset type
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, size))
model = ConvLearner.pretrained(arch, data, precompute=True)
model.fit(0.01, 2)


