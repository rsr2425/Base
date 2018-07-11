## Base
Base is a support tool created for the fastai deep learning library.
The goal of this tool is to minimize the amount of effort needed to get started with image classification.

This library allows the user to download images for training and validation using the bing image library (apparently Google doesn't have an API so unforntuantely Bing is what we're stuck with).
The tool downloads according to the specified search terms and then builds out a folder structure at the specified path compatible for use with the fastai library.

## Getting Started
Unfortunately, you'll need to create an account with Bing in order to use the API.  There are several tiers-----need to fill in-----

## Instructions

## Usage
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


   

## Attribution
Not all of the code in this project is mine.
I also relied on several tutorils in order to put this project together.
These include:
------