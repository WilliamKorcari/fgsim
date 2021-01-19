import uproot
import numpy as np
import awkward as ak
from typing import List
from .cli import args
from collections import OrderedDict

fn = f"wd/{args.tag}/train/ntupleTree.root"
rf = uproot.open(fn)


def ga(vars: List[str], library: str = "ak") -> np.ndarray:
    valvars = rf["/treeMaker;1/tree;1"]
    outpd = valvars.arrays(*[vars], library=library)
    return outpd


xyz = ["recHit_x", "recHit_y", "recHit_z"]
tmparrD = ga(xyz)
posD = OrderedDict([(v, tmparrD[v]) for v in xyz])
del tmparrD

# This yield a dict with 1dim ak arrays:
# posD={"x": [
#     [x_{event=0,entry=0},x_{01},x_{02}],
#     [x_{event=1,entry=0},x_{11},x_{12}],
#     ]...}

# Map negativ hits back to the positive axis
# posD["recHit_z"] = np.abs(posD["recHit_z"])

# Now we need to combine these arrays to a shape with
#  the from Nevents X hits(variable)  X coordinates(3)

# Awkward array handling

# ak.zip({"x": ak.Array([[1], [2, 4, 1], [3]]), "y": ak.Array([[4], [5, 4, 1], [6]])})
# <Array [[{x: 1, y: 4}], ... [{x: 3, y: 6}]] type='3 * var * {"x": int64, "y": in...'>
# hitarr = ak.zip({v:posD[v][0] for v in xyz})
#
hitarr = ak.zip([posD[v][0] for v in xyz])
