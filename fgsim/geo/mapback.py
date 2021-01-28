# from multiprocessing import Pool

# Uproot4 is yet to implement writing...
import awkward as ak
import numpy as np
import yaml

# from numba import jit
from tqdm import tqdm
from multiprocessing import Pool

from ..config import conf
from ..utils import timing_val


def get_idxs_under_threshold(arr: np.array):
    arr = np.argwhere(np.abs(arr) < conf["mapper"]["energy_threshold"]).T
    return arr[:min(len(arr), 5000)]


class mapBack:
    def __init__(self) -> None:
        with open(f"wd/{conf.tag}/cellpos.yaml", "r") as f:
            y = yaml.load(f, Loader=yaml.SafeLoader)
            self.cellposD = {v: np.array(y[v]) for v in conf["mapper"]["xyz"]}
        self.xyz = conf["mapper"]["xyz"]

    @timing_val
    def _map_calo_to_hitsMListDict(self, eventNumber: int, caloimg) -> ak.Array:
        print(f"Start on event {eventNumber}")
        # Get the indices with entries over the energy threshold
        # idxs=np.array(np.where(abs(caloimg) < conf["mapper"]["energy_threshold"])).T
        idxs = get_idxs_under_threshold(caloimg)
        outD = {}
        outD["eventNumber"] = np.repeat(eventNumber, len(idxs))
        for coordinate in self.xyz:
            outD[coordinate] = []
        outD["energy"] = []

        # for idx in tqdm(idxs):
        for idx in idxs:
            for coordinate, i in zip(self.xyz, idx):
                outD[coordinate].append(self.cellposD[coordinate][i])
            outD["energy"].append(caloimg[tuple(idx)].item())
        return ak.Array(outD)

    @timing_val
    def _map_calo_to_hitsMbuilderRec(self, eventNumber: int, caloimg) -> ak.Array:
        idxs = get_idxs_under_threshold(caloimg)
        builder = ak.ArrayBuilder()
        for idx in idxs:
            builder.begin_record()
            for coordinate, i in zip(self.xyz, idx):
                builder.field(coordinate)
                builder.real(self.cellposD[coordinate][i])

            # idx is a tensor here, to get the positon, it needs to be converted to a tuple
            builder.field("energy")
            builder.real(caloimg[tuple(idx)].item())
            builder.field("eventNumber")
            builder.integer(eventNumber)
            builder.end_record()

        return ak.zip(builder.snapshot())
    
    @timing_val
    def _map_calo_to_hitsMbuilderList(self, eventNumber: int, caloimg) -> ak.Array:
        idxs = get_idxs_under_threshold(caloimg)
        builder = ak.ArrayBuilder()

        builder.begin_record("eventNumber")
        builder.begin_list()
        for idx in idxs:
            builder.append(eventNumber)
        builder.end_list()
        builder.end_record()

        for coordinate in set(self.xyz):
            builder.begin_record(coordinate)
            builder.begin_list()
            for idx in idxs:
                builder.append(self.cellposD[coordinate][idx])
            builder.end_list()
            builder.end_record()

        builder.begin_record("energy")
        builder.begin_list()
        for idx in idxs:
            builder.append(caloimg[tuple(idx)].item())
        builder.end_list()
        builder.end_record()

        return builder.snapshot()

    def map_events(self, events) -> ak.Array:
        # builder = ak.ArrayBuilder()
        # for i in range(len(events)):
        #     builder.begin_list()
        #     self._map_calo_to_hits(i, events[i], builder)
        #     builder.end_list()
        # arr = builder.snapshot()
        collD = {"eventNumber": [], "energy": []}
        for coordinate in self.xyz:
            collD[coordinate] = []

        nevents = len(events)
        with Pool(5) as p:
            # r = list(
            #     tqdm(
            #         p.starmap(
            #             self._map_calo_to_hits,
            #             zip(range(nevents), events),
            #         ), total=nevents
            #     )
            # )
            r = list(
                p.starmap(
                    self._map_calo_to_hits,
                    zip(range(nevents), events),
                )
            )
        for fct in (self._map_calo_to_hitsMListDict, self._map_calo_to_hitsMbuilderList, self._map_calo_to_hitsMbuilderRec):
            print(fct.__name__)
            for eventNumber, caloimg in zip(range(nevents), events):
                arr = fct(eventNumber, caloimg)
                break
        
        # r is a list( ak arrays ({hit_x: [] }) )
        raise Exception
        # Collect the values in columns
        collD = {key: ak.concatenate([ev[key] for ev in r]) for key in collD}
        # for i in range(len(events)):
        #     iD = self._map_calo_to_hits(i, events[i])
        #     for key in iD:
        #         collD[key].append(iD[key])
        # for key in collD:
        #     collD[key] = ak.concatenate(collD[key])
        res = ak.array(collD)
        pass
        return res
