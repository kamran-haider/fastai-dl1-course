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
# build the model
model = ConvLearner.pretrained(arch, data, precompute=True)
# train the model
# the model is trained on the training set
model.fit(0.01, 2)

# Now that the model is trained, we can take a look at its performance
# we do this by checking results on the validation set. But first we need 
model.predict()