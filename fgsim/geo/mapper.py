import awkward as ak
import numpy as np
from collections import OrderedDict
from pprint import pprint


class Geomapper:
    """
    This class provides a geomapper object, that is used to map hits ton cells.
    """

    def __init__(self, posD: OrderedDict):
        self.xyz = posD.keys()
        # reduce the number of entries
        for v in self.xyz:
            posD[v] = self.__redArray(posD[v])
        self.flatposD = posD
        del posD
        self.dist_threshold = 1e-02

        binbordersD = OrderedDict()
        for v in self.xyz:
            binbordersD[v] = self._providemapping(self.flatposD[v])

    def __redArray(self, arr: ak.Array):
        # limit the number of events condidered to calculate the binning to 10 events
        arr = arr[: min(len(arr), 10)]
        return ak.flatten(ak.Array(arr))

    def _providemapping(self, arr: ak.Array):
        # Convert to numpy array
        arr = ak.to_numpy(arr, allow_missing=False)

        # the unique entries will later corrispond
        # to the lower border of the binning
        uniques = np.unique(arr)

        # calculate the distance
        dist = uniques[:-1] - uniques[1:]

        # remove the distances under a thereshold
        idx_to_del = []
        for i, d in enumerate(dist):
            if abs(d) < self.dist_threshold:
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

        print(f"hval prev: {hval}")
        # first all vales in hval that are close to each other are collected in one
        compare_idx, running_idx = 0, 1
        while compare_idx < len(hval) - 1:
            assert compare_idx != running_idx
            if np.isclose(
                hval[compare_idx], hval[running_idx], rtol=self.dist_threshold
            ):
                del hval[running_idx]
            else:
                running_idx = running_idx + 1
                if running_idx == len(hval):
                    compare_idx = compare_idx + 1
                    running_idx = compare_idx + 1
        print(f"hval post: {hval}")

        # Next the we iterate over the distances
        # We check if the distances are close to the most frequent values and if so the distance is replace
        for idx, d in enumerate(dist):
            for val in hval:
                if np.isclose(d, val, rtol=self.dist_threshold):
                    dist[idx] = val
                    continue


        # throw singles out
        # dist_counts = np.unique(dist, return_counts=1)
        # counts_required = max(dist_counts[1]) / 4
        # low_count_filter = filter(lambda x: x[1] < counts_required, zip(*dist_counts))
        # infreq, infreq_cnt = zip(*low_count_filter)

        # print("New var")
        # pprint({"dist": dist, "hval": dist_counts[0], "hcnt": dist_counts[1]})



        hval, hcnt = np.sort(np.unique(dist, return_counts=1))[..., ::-1]
        pprint({"dist": dist, "hval": hval, "hcnt": hcnt})


        dist_counts = np.unique(dist, return_counts=1)
        counts_required = max(hcnt) / 4

        low_count_filter = filter(lambda x: x[1] < counts_required, zip(*dist_counts))
        infreq, infreq_cnt = zip(*low_count_filter)

        highcount_filter = filter(lambda x: x[1] >= counts_required, zip(*dist_counts))
        freq, freq_cnt = zip(*highcount_filter)

        infreq_idxs = [idx for idx, val in enumerate(dist) if val in infreq]

        return uniques

    def _getpixel(val, var: str):
        return np.digitize(
            val,
        )

    def map(arr: ak.Array):
        return np.array([])
