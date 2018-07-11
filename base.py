'''
Module: base

Contains the ImBase class which helps a user search for images on internet.
Images are then downloaded and split into training/validation sets.

Bing API key should be stored in a python file called secret.

'''
from secret import APIKEY



class ImBase(object):
    def __init__(self, cls, sts, pth, mr=50, gs=250, train_splt=0.7):
        self.classes = cls
        self.searchterms = sts
        self.path = pth
        self.max_results = 50
        self.group_size = 50
        self.train_splt = 0.7

    def run(self):
        pass

    def download(self, cl, st):
        '''
        Downloads images and places ----------
        :param cl:
        :param st:
        :return:
        '''