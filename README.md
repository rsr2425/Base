## Base
Base is a support tool created for the fastai deep learning library.
The goal of this project is to minimize the amount of effort needed to get started with image classification.
Using Base, it'll become much easier to build your own datasets to build classifcation 
tools with by using images downloaded from the internet (inspired by this blog 
post --------------, in which the author builds his own pokedex!)

This library allows the user to download images for training and validation using the bing image library
 (apparently Google doesn't have an API so unforntuantely Bing is what we're stuck with).
The tool downloads according to the specified search terms and then builds out a folder structure at the specified path compatible for use with the fastai library.

## Getting Started
Unfortunately, you'll need to create an account with Bing in order to use their API.
There are several tiers, one free and several paid as well as a free trial.  You 
can find more information here:

-----need to fill in-----sdfsdf

## Usage

When using this tool, make sure that you have a secret.py file in the same directory.
This file should contain your api key (stored as APIKEY) as well as your path information.
It is separated out so that this information is not accidentally pushed to github.


The following code snippet demonstrates how to use the Base library with fastai.
We'll try to use the fastai library to classify pokemon based on how they look.

    # first section creates the code
    import base

    classes = ['fire', 'water', 'thunder']
    search_terms = ['fire pokemon and charmander', 'water pokemon',
                    'thunder pokemon']

    # downloads images for each class in ~/data
    downloader = base.ImBase(classes, search_terms, "~/data")
    downloader.run()

    # now that our images our downloaded, we can use the fastai library
    # as it was used in lesson 1
    # this assumes you've already done the imports needed
    arch = resnet34
    data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
    learn = ConvLearner.pretrained(arch, data, precompute=True)
    learn.fit(0.01, 2)

-------NOTE SOMEWHERE THAT CLASS NAMES CAN ONLY BE A SINGLE WORD----

   

## Attribution
Not all of the code in this project is mine.
I also relied on several tutorials in order to put this project together.
These include:
------