# Uproot4 is yet to implement writing...
import awkward as ak
import uproot3 as uproot
import numpy as np

from .config import conf


def dump_generated_events(arr: ak.Array):

    fn = f"wd/{conf.tag}/output.root"
    n = len(arr)
    with uproot.recreate(fn) as file:
        file["tree1"] = uproot.newtree(dict(arr.type.type))
        file["tree1"].extend(
            {branch: arr[branch] for branch in arr.fields}
        )
