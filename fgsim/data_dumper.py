# Uproot4 is yet to implement writing...
import awkward as ak
import uproot3 as uproot

from .config import conf

def dump_generated_events(arr: ak.Array):

    fn = f"wd/{conf['tag']}/output.root"
    with uproot.recreate(fn) as file:
        file["tree1"] = uproot.newtree()
        for branch in arr.fields:
            ak0_array = ak.to_awkward0(arr[branch])
            n = arr[branch].counts
            file["tree1"].extend({branch: ak0_array, "n": n})
