from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
from PIL import Image
import requests
from io import BytesIO
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import re
import sys
import tarfile
import numpy as np
from six.moves import urllib
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow as tf

import urllib.request
from urllib.request import Request, urlopen
import requests
import shutil
import re, math
from collections import Counter
from scipy import spatial
import gensim