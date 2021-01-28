from collections import OrderedDict
from multiprocessing import Pool
from typing import Tuple

import awkward as ak
import numpy as np
import yaml

from ..config import conf

# from ..plot import plot3d, plot_z_pos

# from pprint import pprint
# from itertools import combinations


class Geomapper:
    """
    This class provides a geomapper object, that is used to map hits ton cells.
    """

    def __init__(self, posD: OrderedDict):
        # reduce the number of entries
        for v in conf.mapper.xyz:
            posD[v] = self.__redArray(posD[v])
        self.flatposD = posD

        self.cells_on_axis = OrderedDict()
        for v in conf.mapper.xyz:
            self.cells_on_axis[v] = self.__compute_cells_on_axis(self.flatposD[v])
        with open(f"wd/{conf.tag}/cellpos.yaml", "w") as f:
            yaml.dump(
                {v: self.cells_on_axis[v].tolist() for v in self.cells_on_axis},
                f,
                Dumper=yaml.SafeDumper,
            )

        self.binbordersD = {
            v: self.__cellaxis_to_binborders(self.cells_on_axis[v])
            for v in conf.mapper.xyz
        }
        # with open(f"wd/{conf.tag}/binbordes.yaml", "w") as f:
        #     yaml.dump(
        #         {v: self.binbordersD[v].tolist() for v in self.binbordersD},
        #         f,
        #         Dumper=yaml.SafeDumper,
        #     )
        # setup the empty image of the caloriment to copy
        self.emptycaloarr = np.zeros(
            [len(self.binbordersD[v]) + 1 for v in conf.mapper.xyz]
        )

        # Debug code
        # arr = [ak.to_numpy(posD[v]) for v in conf.mapper.xyz]
        # plot3d(arr)
        # foo = np.column_stack(arr)
        # from ..utils import most_freq_zval
        # plot_z_pos(foo, most_freq_zval(posD))

    def __cellaxis_to_binborders(self, cellpostions: np.ndarray) -> np.ndarray:
        # the binborders here are dont include a upper or lower bound
        # because np.digitalize will map all positions that are under the
        # first bin border and vice versa for the maximum.
        binborder = (cellpostions[1:] + cellpostions[:-1]) / 2

        # assure they are increasing
        assert all(np.diff(binborder) > 0)
        return binborder

    def __redArray(self, arr: ak.Array) -> ak.Array:
        # limit the number of events condidered to calculate the binning to 10 events
        arr = arr[: min(len(arr), 10)]
        return ak.flatten(ak.Array(arr))

    # This returns the grid of the cells anlong the given axis
    def __compute_cells_on_axis(self, arr: ak.Array) -> np.ndarray:
        # Convert to numpy array
        arr = ak.to_numpy(arr, allow_missing=False)

        # the unique entries will later corrispond
        # to the lower border of the binning
        uniques = np.unique(arr)
        # print(f"uniques {uniques} ")
        # calculate the distance
        dist = uniques[1:] - uniques[:-1]

        # remove the distances under a thereshold
        idx_to_del = []
        for i, d in enumerate(dist):
            if abs(d) < conf.mapper.threshold:
                idx_to_del.append(i)
        dist = np.delete(dist, idx_to_del)
        uniques = np.delete(uniques, [i + 1 for i in idx_to_del])

        # Count the frequency of distances and short by frequency
        # np.unique: get unique values and the counts
        # np.sort: sort the array by the lasts axis / the counts
        # [...,::-1] to start with the highest element
        tmparr = np.sort(np.unique(dist, return_counts=1))[..., ::-1]
        # [0] only get the values, not the counts
        # Convert to a list to make removing elements less painful
        hval = list(tmparr[0])
        del tmparr

        # first all vales in hval that are close to each other are collected in one
        compare_idx, running_idx = 0, 1
        while compare_idx < len(hval) - 1:
            assert compare_idx != running_idx
            if np.isclose(
                hval[compare_idx], hval[running_idx], rtol=conf.mapper.threshold
            ):
                del hval[running_idx]
            else:
                running_idx = running_idx + 1
                if running_idx == len(hval):
                    compare_idx = compare_idx + 1
                    running_idx = compare_idx + 1

        # Next the we iterate over the distances
        # We check if the distances are close to the most frequent
        # values and if so the distance is replace
        for idx, d in enumerate(dist):
            for val in hval:
                if np.isclose(d, val, rtol=conf.mapper.threshold):
                    dist[idx] = val
                    continue

        sum = min(uniques)
        cellpositions = [sum]
        for val in dist:
            sum += val
            cellpositions.append(sum)
        # print(f"dist {dist} \n \n cellpositons \n {cellpositions}")
        cellpositions = np.array(cellpositions)
        assert all(np.diff(cellpositions) > 0)
        return cellpositions

        # hval, hcnt = count_and_sort(dist)
        # pprint({"dist": dist, "hval": hval, "hcnt": hcnt})

        # # Time for the combinatorics:
        # # check if we can replace some of the the less frequent
        # # distances by the sum of  the two more frequent one
        # # this is to. This avoids that we miss cells by accident
        # dist_counts = np.unique(dist, return_counts=1)
        # counts_required = max(hcnt) / 3

        # low_count_filter = filter(lambda x: x[1] < counts_required, zip(*dist_counts))
        # infreq, infreq_cnt = zip(*low_count_filter)

        # highcount_filter = filter(lambda x: x[1] >= counts_required, zip(*dist_counts))
        # freq, freq_cnt = zip(*highcount_filter)

        # infreq_idxs = [idx for idx, val in enumerate(dist) if val in infreq]

        # pairs_of_freq = list(combinations(freq, 2))
        # val_to_pairD = {a + b: (a, b) for a, b in pairs_of_freq}

        # mapInfreqD = {}
        # for infreqval in infreq:
        #     for freqval, pair in val_to_pairD.items():
        #         if np.isclose(infreqval, freqval, conf.mapper.threshold):
        #             mapInfreqD[infreqval] = pair
        #             break

    def __getpixel(self, val: ak.Array, var: str) -> np.int:
        val = ak.to_numpy(val)
        border = self.binbordersD[var]
        pos = np.digitize(val, border) - 1
        return pos

    def __map_hit_to_idx(self, hit: ak.Array) -> Tuple:
        idx = tuple((self.__getpixel(hit[v], v) for v in conf.mapper.xyz))
        return idx

    def _map_event_to_calo(self, hitsA: ak.Array) -> np.ndarray:
        calo = self.emptycaloarr.copy()
        for hit in hitsA:
            idx = self.__map_hit_to_idx(hit)
            assert len(idx) == len(calo.shape)
            calo[idx] += 1
        return calo

    def map_events(self, eventarr: ak.Array) -> np.ndarray:
        # caloimgL = [self.__map_event_to_calo(hitsA) for hitsA in eventarr]
        with Pool(5) as p:
            caloimgL = p.map(self._map_event_to_calo, eventarr)

        # if (
        #     len(np.unique(caloimgL[0][3, :, 4])) == 1
        #     and np.unique(caloimgL[0][3, :, 4])[0] != 0
        # ):
        #     raise Exception
        # from ..utils import most_freq_zval
        # zidx = np.digitize(most_freq_zval(self.flatposD), self.binbordersD["recHit_z"])
        # writeimage(caloimgL[0][..., zidx], f"wd/{conf.tag}/layer{zidx}.png")
        return np.stack(caloimgL, axis=0)


def count_and_sort(arr) -> np.ndarray:
    return np.sort(np.unique(arr, return_counts=1))[..., ::-1]


def writeimage(arr: np.ndarray, fn: str = f"wd/{conf.tag}/out.png"):
    import matplotlib.pyplot as plt

    plt.matshow(arr)
    plt.savefig(fn)
