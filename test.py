'''
Module: test

Testing module intended to be used with PyTest.
'''

from base import ImBase

import sys

# fast test
if int(sys.argv[1]) == 1:
    cls = sts = ['bulbasaur']
    downloader = ImBase(cls, sts, gs=50, p=False)
    downloader.run()

# extended test
elif int(sys.argv[1]) == 2:
    cls = sts = ['bulbasaur', 'charmander', 'squirtle']
    downloader = ImBase(cls, sts)

    downloader.run()