from collections import OrderedDict
from typing import List

import os
import pickle
import awkward as ak
import numpy as np
import uproot

from ..config import conf

# load the root table
fn = conf["luppath"]
rf = uproot.open(fn)
arr = rf["analyzer/tree"].arrays()
keydf = ak.to_pandas(arr[0])


# load the geometry

fngeopic = conf["geoyamlpath"].strip('yaml')+'pickle'
if os.path.isfile(fngeopic):
    with open(fngeopic, "rb") as f:
        geoD = pickle.load(f)
else:
    with open(geoyamlpath, "r") as f:
        geoD = yaml.load(f)
    with open(fngeopic, "wb") as f:
        pickle.dump(geoD, f)
