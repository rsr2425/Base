'''
Module: base

Contains the ImBase class which helps a user search for images on internet.
Images are then downloaded and split into training/validation sets.

Bing API key should be stored in a python file called secret.

'''
import numpy as np
import pandas as pd

from requests import exceptions
import multiprocessing as mp
import requests
import cv2
import shutil
import os

from random import shuffle
from math import floor

from secret import APIKEY, BASE

# set the endpoint API URL
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

# when attempting to download images from the web both the Python
# programming language and the requests library have a number of
# exceptions that can be thrown so let's build a list of them now
# so we can filter on them
EXCEPTIONS = set([IOError, FileNotFoundError, exceptions.RequestException, exceptions.HTTPError,
                  exceptions.ConnectionError, exceptions.Timeout])


def batches(l: list, n: int):
    '''
    Breaks down list into n-sized chunks
    :param l: list to be broken down
    :param n: size for each chunk from list
    :return: iterator over chunks
    '''
    for i in range(0, len(l), n):
        yield l[i:i+n]

class ImBase(object):
    def __init__(self, cls, sts, out_f='data', mr=50, gs=250, ts=0.7
                 , p=True):
        self.classes = cls
        self.searchterms = sts
        self.output_fldr = out_f
        self.max_results = mr
        self.group_size = gs
        self.train_splt = ts
        self.parallel = p

    def run(self):
        work = list(zip(self.classes, self.searchterms))
        if self.parallel:
            num_workers = mp.cpu_count()
            pool = mp.Pool(num_workers)

            for batch in batches(work
                             , num_workers):
                for cl, st in batch:
                    re = pool.apply_async(self.download, args=(cl,st,))
                pool.close()
                pool.join()
        else:
            for cl, st in work:
                self.download(cl, st)

    def download(self, cl, st):
        '''
        Downloads images and places them in train/valid folders at self.path.
        :param cl: String of class name
        :param st: String of search term used on Bing Image Search.
        :return:
        '''
        # output_fldr is initial location for files after downloading
        # if empty string is passed for output, the temp folder will be used
        # output_fldr does not exist, create it
        output_dir = os.path.join(BASE, self.output_fldr)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        # store the search term in a convenience variable then set the
        # headers and search parameters
        headers = {"Ocp-Apim-Subscription-Key": APIKEY}
        params = {"q": st, "offset": 0, "count": self.group_size}


        # make the search
        print("[INFO] searching Bing API for '{}'".format(st))
        search = requests.get(URL, headers=headers, params=params)
        search.raise_for_status()

        # grab the results from the search, including the total number of
        # estimated results returned by the Bing API
        results = search.json()
        estNumResults = min(results["totalEstimatedMatches"]
                            , self.max_results)
        print("[INFO] {} total results for '{}'".format(estNumResults,
                                                        st))
        # initialize the total number of images downloaded thus far
        total = 0

        # output_fldr is initial location for files after downloading
        # if empty string is passed for output, the temp folder will be used
        # output_fldr does not exist, create it
        output_fldr = 'tmp'
        output_dir = os.path.join(BASE, output_fldr)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # loop over the estimated number of results in `GROUP_SIZE` groups
        for offset in range(0, estNumResults, self.group_size):
            # update the search parameters using the current offset, then
            # make the request to fetch the results
            print("[INFO] making request for group {}-{} of {}...".format(
                offset, offset + self.group_size, estNumResults))
            params["offset"] = offset
            search = requests.get(URL, headers=headers, params=params)
            search.raise_for_status()
            results = search.json()
            print("[INFO] saving images for group {}-{} of {}...".format(
                offset, offset + self.group_size, estNumResults))

            # loop over the results
            for v in results["value"]:
                # try to download the image
                try:
                    # make a request to download the image
                    print("[INFO] fetching: {}".format(v["contentUrl"]))
                    r = requests.get(v["contentUrl"], timeout=30)

                    # build the path to the output image
                    ext = v["contentUrl"][v["contentUrl"].rfind("."):]
                    score = os.path.sep.join([output_dir,
                                              f"{cl}_{str(id)}_{str(total).zfill(8)}{ext}"])

                    # write the image to disk
                    f = open(score, "wb")
                    f.write(r.content)
                    f.close()

                # catch any errors that would not unable us to download the
                # image
                except Exception as e:
                    # check to see if our exception is in our list of
                    # exceptions to check for
                    if type(e) in EXCEPTIONS:
                        print("[INFO] skipping: {}".format(v["contentUrl"]))
                        continue

                # try to load the image from disk
                image = cv2.imread(score)

                # if the image is `None` then we could not properly load the
                # image from disk (so it should be ignored)
                if image is None:
                    print("[INFO] deleting: {}".format(score))
                    os.remove(score)
                    continue

                # update the counter
                total += 1

    def shuffle(self):
        pass