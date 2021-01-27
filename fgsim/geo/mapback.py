from multiprocessing import Pool

# Uproot4 is yet to implement writing...
import awkward as ak
import numpy as np
import yaml

from ..config import conf


class mapback:
    def __init__(self) -> None:
        with open(f"wd/{conf.tag}/binbordes.yaml", "r") as f:
            y = yaml.load(f, Loader=yaml.SafeLoader)
            self.binbordersD = {v: np.array(y[v]) for v in self.binbordersD}
        self.xyz = conf["mapper"]["xyz"]

    def mapImage(self, eventid: int, caloimg) -> ak.Array:
        hitsL = []
        for idx in np.nonzero(caloimg):
            d = {
                coordinate: self.binbordersD[coordinate][i]
                for coordinate, i in zip(self.xyz, idx)
            }
            d["energy"] = caloimg[idx]
            d["eventNumber"] = eventid
            hitsL.append(ak.Array(d))
        return ak.Array(hitsL)

    def mapEvents(self, events) -> ak.Array:
        eventsL = []
        for i in range(len(events)):
            eventsL.append(self.mapImage(i, events[i]))
        arr = ak.Array(eventsL)
        return ak.zip(arr)
